#!/usr/bin/env python3
"""Internal JSON-lines worker; stdout is reserved for the public protocol."""
from __future__ import annotations
import contextlib
from functools import lru_cache, partial
import json
import math
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
if __name__ == '__main__':  # the clients run this file as a script: import the exhad package from ROOT
    sys.path[0] = str(ROOT)
from exhad.models import SCALARS  # noqa: E402


@lru_cache(maxsize=3)
def _conditional_pool(binary, portal):
    from exhad.accelerator import sampler
    pools = {'alp-fermion': sampler.AlpFermionAccelerator, 'scalar': sampler.ScalarAccelerator, 'b-l': sampler.VectorAccelerator}
    return pools[portal](ROOT / binary)


def sample(config, request):
    execution = request.get('execution', 'auto')

    if execution not in {'auto', 'reference'}:
        raise ValueError('Unknown execution mode')

    if 'rows' in request:
        from exhad.exclusive import sample_rows
        return sample_rows(config, request)

    if request.get('all_decays', False) or (request.get('model') == 'hnl' and request.get('mixing') is not None
                                            and 'primary_events' not in request):
        from exhad.full_decays import sample_all
        result = sample_all(config, request, sample)
        return result if request.get('all_decays', False) else result['events']
    from exhad.exclusive import sample_hadronic, start_mass

    if float(request['mass']) < start_mass(request['model']):
        return sample_hadronic(config, request)
    matched = _sample_reference

    if execution == 'auto' and not request.get('weighted', False):
        model, mass = request['model'], float(request['mass'])
        portal = ('alp-fermion' if model == 'alp-fermion' and mass > 2.4 else 'scalar' if model in SCALARS and 3. <= mass <= 5.
                  else 'b-l' if model == 'b-l' and 2. <= mass <= 5. else None)
        if portal is not None:
            matched = partial(_sample_reference, active_pool=_conditional_pool(config['conditional_acceleration'], portal))

    if (request['model'] == 'alp-fermion' or request['model'] in SCALARS) and float(request['mass']) > 4.:
        from exhad.transition import sample as transition
        return transition(config, request, matched)

    return matched(config, request)


def _sample_reference(config, request, active_pool=None):
    from exhad import registry
    from exhad.model1.sampler import sample_deployment
    model = request['model']

    if request.get('weighted', False):
        if model != 'alp-fermion':
            raise ValueError('Weighted sampling supports only alp-fermion')
        result = sample_deployment(model, request['mass'], request['events'],
            seed=request['seed'], variation=request['variation'], weighted=True,
            weight_floor_fraction=request.get('weight_floor_fraction', 0.))
        result['events'] = [[list(map(float, row[i:i + 6])) for i in range(0, len(row), 6)
                            if int(row[i + 5]) != -999] for row in result['events']]
        return result

    if model == 'hnl':
        from exhad.hnl import hadronize
        return hadronize(config, request)
    mass, count, seed = request['mass'], request['events'], request['seed']
    variation = request['variation']

    if model == 'dark-photon':
        rows = registry.dark_photon_events(mass, count, seed, None if variation == 'central' else variation)
    elif model == 'b-l':
        rows = registry.b_l().sample(mass, count, seed=seed, variation=variation,
                                             active_pool=active_pool).events
    elif model == 'alp-fermion' or model in SCALARS:
        rows = sample_deployment(model, mass, count, seed=seed, variation=variation, active_pool=active_pool)
    else:
        raise ValueError(f'Unknown model {model!r}')

    return [[list(map(float, row[i:i + 6])) for i in range(0, len(row), 6)
             if int(row[i + 5]) != -999] for row in rows]


def main(cpp=False):
    config = json.loads((ROOT / '.runtime/current.json').read_text())

    if 'conditional_acceleration' not in config:
        raise SystemExit('exHad is not configured; run python tools/configure.py')
    os.environ['EXHAD_RUNTIME_BINARY'] = str(ROOT / 'cpp/exhad')
    os.environ['EXHAD_PYTHIA8DATA'] = config['xmldoc']
    protocol = sys.stdout
    # A Generator worker owns this process; request-seeded Pythia runtimes persist across batches.
    # For the C++ client it instead serves each request through a Generator of its worker processes.
    generators = {}

    for line in sys.stdin:
        try:
            with contextlib.redirect_stdout(sys.stderr):
                result = cpp_sample(generators, cpp_request(line)) if cpp else sample(config, json.loads(line))
            text = serialize(result if isinstance(result, dict) else {'events': result}, cpp)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            error = f'{type(exc).__name__}: {exc}'
            text = 'ERROR ' + ' '.join(error.split()) + '\n' if cpp else json.dumps({'error': error}) + '\n'
        protocol.write(text)
        protocol.flush()

    for generator in generators.values():
        generator.close()


def cpp_sample(generators, request):
    from exhad import Generator
    key = tuple(request.pop(name) for name in ('model', 'variation', 'execution', 'workers', 'chunk_size'))

    if key not in generators:
        generators[key] = Generator(key[0], variation=key[1], execution=key[2], workers=key[3], chunk_size=key[4])
    mass, events, seed = request['mass'], request['events'], request['seed']

    if 'rows' in request:  # the labels keep the order the client sent them in
        realized = generators[key].generate_rows(mass, request['rows'], seed=seed,
                                                 terminal=request['terminal'], mixing=request['mixing'])
        return [event for label in request['rows'] for event in realized[label]]

    if request['all_decays']:
        return generators[key].generate_all(mass, events, seed=seed, mixing=request['mixing'])

    return generators[key].generate(mass, events, seed=seed, mixing=request['mixing'])


def serialize(reply, cpp):
    """The complete reply text, once every particle row (six finite fields, E >= 0, m >= 0,
    integral nonzero PDG) and importance weight is valid."""

    if not all(len(p) == 6 and all(map(math.isfinite, p)) and min(p[3], p[4]) >= 0 and p[5] == int(p[5]) != 0
               for event in reply['events'] for p in event) or not all(map(math.isfinite, reply.get('raw_weights', ()))):
        raise ValueError('invalid particle row or importance weight in reply')

    if not cpp:
        return json.dumps(reply, allow_nan=False) + '\n'
    rows = [f'OK {len(reply["events"])}']

    for event in reply['events']:
        rows += [f'EVENT {len(event)}', *(' '.join(format(x, '.17g') for x in particle) for particle in event)]

    return '\n'.join(rows + ['END\n'])


def cpp_request(line):
    from exhad.models import validate_model
    model, variation, execution, workers, chunk_size, mass, count, seed, scope, ue, umu, utau, *extra = line.split()
    validate_model(model)
    workers, chunk_size, mass, count, seed = int(workers), int(chunk_size), float(mass), int(count), int(seed)

    if not 0 <= workers <= 64 or not 1 <= chunk_size <= 1000000:
        raise ValueError('Invalid worker count or chunk size')

    if not math.isfinite(mass) or mass <= 0 or not 0 <= count <= 1000000 or not 0 <= seed < 2**64:
        raise ValueError('Invalid mass, event count or seed')

    if scope not in {'all', 'hadronic', 'rows'} or bool(extra) != (scope == 'rows'):
        raise ValueError('Unknown decay scope or trailing request fields')
    mixing = list(map(float, (ue, umu, utau)))

    if any(not math.isfinite(x) or x < 0 for x in mixing) or (model == 'hnl' and sum(mixing) <= 0):
        raise ValueError('Invalid squared HNL mixings')
    request = dict(model=model, variation=variation, execution=execution, workers=workers or None,
                   chunk_size=chunk_size, mass=mass,
                   events=count, seed=seed, all_decays=scope == 'all',
                   mixing=mixing if model == 'hnl' else None)

    if scope == 'rows':  # ... rows ue umu utau TERMINAL LABEL=COUNT [LABEL=COUNT ...]
        rows = dict((label, int(n)) for label, n in (pair.split('=') for pair in extra[1:]))
        if len(rows) != len(extra) - 1 or any(n < 0 for n in rows.values()) or sum(rows.values()) != count:
            raise ValueError('Invalid row counts')
        request.update(rows=rows, terminal=extra[0])

    return request


if __name__ == '__main__':
    main(cpp=sys.argv[1:] == ['--cpp'])
