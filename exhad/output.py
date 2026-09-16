"""Portable event records, independent of the simulation consuming them."""
import json
import math
from pathlib import Path
from .models import mother_pdg as resolve_mother_pdg


def check_format(format):  # pyhepmc for HepMC, None for JSON
    if format not in {'json', 'hepmc', 'hepmc2'}:
        raise ValueError('format must be json, hepmc, or hepmc2')

    if format == 'json':
        return None

    try:
        import pyhepmc
    except ImportError as exc:
        raise RuntimeError('HepMC output requires pyhepmc, a dependency of exHad: '
                           'python -m pip install pyhepmc') from exc

    return pyhepmc


def write_events(path, payload, *, format='json', mother_pdg=None):
    """Write JSON or HepMC3/HepMC2 ASCII without overwriting a file.

    HepMC has one rest-frame parent (status 2), one vertex at the origin,
    and the returned final particles (status 1). It is a collapsed decay
    record, not a reconstructed history of intermediate resonances.
    The mother PDG identifier is mother_pdg when given, else the default of
    payload['model'] (exhad.models.MOTHER_PDG).
    """
    path = Path(path)
    if format in {'hepmc', 'hepmc2'}:
        mother_pdg = resolve_mother_pdg(payload['model'], mother_pdg)
    pyhepmc = check_format(format)
    if pyhepmc is None:
        with path.open('x') as stream:
            json.dump(payload, stream, allow_nan=False)
            stream.write('\n')
        return
    mass = float(payload['mass_gev'])
    if not math.isfinite(mass) or mass <= 0:
        raise ValueError('mother mass must be finite and positive')
    events = payload['events']
    if 'raw_weights' in payload and 'hadronization_weights' not in payload:
        raise ValueError('Normalize raw importance weights over the complete sample before HepMC export')
    weights = payload.get('hadronization_weights', [1.] * len(events))
    if len(weights) != len(events) or any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError('event weights must be finite, positive, and match the event count')
    for particles in events:
        for row in particles:
            if (len(row) != 6 or not all(math.isfinite(x) for x in row)
                    or row[5] != int(row[5]) or not 0 < abs(row[5]) < 2**31):
                raise ValueError('each daughter must contain six finite fields and an integer PDG identifier')
    run = pyhepmc.GenRunInfo()
    run.weight_names = ['nominal']
    run.tools = [('exHad', '', 'Rest-frame decays; intermediate history collapsed')]
    # Exclusive creation also protects against another process creating the
    # destination after the command-line preflight check.
    with path.open('x') as stream:
        with pyhepmc.open(stream, 'w', format='hepmc3' if format == 'hepmc' else 'hepmc2') as writer:
            for number, (particles, weight) in enumerate(zip(events, weights)):
                event = pyhepmc.GenEvent(pyhepmc.Units.GEV, pyhepmc.Units.MM)
                event.run_info = run
                event.event_number = number
                event.weights = [float(weight)]
                vertex = pyhepmc.GenVertex()
                mother = pyhepmc.GenParticle((0., 0., 0., mass), mother_pdg, 2)
                mother.generated_mass = mass
                vertex.add_particle_in(mother)
                for row in particles:
                    particle = pyhepmc.GenParticle(tuple(row[:4]), int(row[5]), 1)
                    particle.generated_mass = row[4]
                    vertex.add_particle_out(particle)
                event.add_vertex(vertex)
                event.attributes['exhad_model'] = str(payload['model'])
                event.attributes['exhad_decay_scope'] = payload.get('decay_scope', 'hadronic')
                writer.write(event)
