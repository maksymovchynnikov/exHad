#!/usr/bin/env python3
"""Build this machine's helper executables and the alp-fermion, scalar and B-L accelerator, and record the build.

Last step of the build, after `make -C cpp`. It builds the accelerator
`.runtime/accelerator/exhad-batch`, whose gates must pass, or
this step fails and leaves the checkout unconfigured: the sha256 pins of the Pythia
files it relies on (headers always; implementation files where a source tree provides
them), the support audit, the enumerated one-step laws against stock draws, pass-through
identity, and the seed-initializer and full-event replays. The accelerator builds against the stock Pythia library.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # import this checkout's exhad package

# The Pythia 8.317 files whose behaviour the accelerator's equivalence argument reads.
# The headers exist in every Pythia installation; the implementation files only in a
# source tree. Neither is patched or compiled here: each is hashed where it is present.
PYTHIA_PINS = {
    'include/Pythia8/FragmentationModel.h': '9d1b67999236d9eb68cba570db02439348cb11240eb6acdac67b52193fd2d3d0',
    'include/Pythia8/UserHooks.h': '283e9d383c899b604fb3e5c47f500d31863e1bfbf05fb1b7f0afe770d9ad081f',
    'include/Pythia8/StringFragmentation.h': '43b39f389ecac321e34a3156e8c7259fbd68ed844d96be679daf00b52c69f4f2',
    'include/Pythia8/FragmentationFlavZpT.h': 'ec3a2ad65584f9e7fe86fb9dfe1af3950a4f856a8ff8ae369e73a96f578b38a5',
    'src/StringFragmentation.cc': 'aa790fc171cf19d7b20f266bbc59ed525356aa881ef72e0668eff8a27fecdecc',
    'src/FragmentationFlavZpT.cc': '60c7a0397f423f2a7e5d434c2e723272c881839ef8feb9e58cfa32b3120e80b9',
    'src/HadronLevel.cc': '86c32d57a558db0e3e6a95ca249fd1770abf6b4b2335106da290c25ae0e475ff',
    'src/Pythia.cc': 'ce92ebd30c439b68cc836f780c6dce249132ff9d9265a8a6fe8fba75dbdd88b9',
    'src/FragmentationModel.cc': '08bc10ee9ebbdb417cf2f68e85f2628c970ed4e64716a8a9a70702aedccfed2c',
}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_pythia_pins(prefix):
    """Hash every pinned file this prefix has; a differing file fails the build."""
    checked = {}

    for name, expected in PYTHIA_PINS.items():
        path = prefix / name
        if not path.is_file():
            if name.startswith('include/'):
                raise ValueError(f'missing Pythia header {path}')
            continue  # an installed Pythia without sources: the behavioural gates still run
        if digest(path) != expected:
            raise ValueError(f'{path} differs from the pinned Pythia 8.317 file '
                             'the accelerator was verified against')
        checked[name] = expected

    return checked


def build_accelerator(out, compiler, includes, link_pythia, prefix, config):
    """Build and gate `exhad-batch`; return its path once every gate passes."""
    from exhad.model1.sampler import load_deployment
    pins = verify_pythia_pins(prefix)
    base = ROOT / 'exhad/accelerator'
    includes = [*includes, '-I' + str(base)]
    link_pythia = [*link_pythia, '-ldl', '-pthread']
    objects = [ROOT / 'cpp/src' / f'{name}.o' for name in ('isospin_cg', 'symmetry_filter')]
    commands = []

    def run(args, **kw):
        args = list(map(str, args))
        commands.append(args)
        return subprocess.run(args, cwd=ROOT, check=True, **kw)

    def compile_all(*jobs):  # independent compiler invocations, four at a time
        jobs = [list(map(str, job)) for job in jobs]
        commands.extend(jobs)
        with ThreadPoolExecutor(4) as pool:
            list(pool.map(partial(subprocess.run, cwd=ROOT, check=True), jobs))

    checks = {mode: out / ('check-rng-' + mode) for mode in ('simd', 'scalar')}
    scalar_rng = '-DEXHAD_RNG_FORCE_SCALAR'
    compile_all([*compiler, *includes, ROOT / 'tests/check_rng_simd.cc', *link_pythia, '-o', checks['simd']],
                [*compiler, scalar_rng, *includes, ROOT / 'tests/check_rng_simd.cc', *link_pythia,
                 '-o', checks['scalar']],
                [*compiler, *includes, '-c', base / 'batch_kernel.cc', '-o', out / 'exhad-batch.o'],
                [*compiler, scalar_rng, *includes, '-c', base / 'batch_kernel.cc', '-o', out / 'exhad-batch-scalar-rng.o'],
                [*compiler, *includes, '-c', base / 'audit_support.cc', '-o', out / 'audit-support.o'])
    compile_all(*([*compiler, out / (binary + '.o'), *objects, *link_pythia, '-o', out / binary]
                  for binary in ('exhad-batch', 'exhad-batch-scalar-rng', 'audit-support')))
    binary, scalar_binary = out / 'exhad-batch', out / 'exhad-batch-scalar-rng'
    # The SIMD seed initializer must reproduce the pinned Pythia RNG, including the scalar fallback.
    rng_checks = {mode: json.loads(run([check], text=True, capture_output=True).stdout)
                  for mode, check in checks.items()}
    # Support, one-step-law and pass-through gates against this very Pythia
    # library and the actual configured central generator settings.
    support = ''
    for portal, models in (('alp-fermion', ('alp-fermion',)), ('scalar', ('scalar-central', 'scalar-lower', 'scalar-upper', 'scalar-1809')),
                           ('b-l', ('b-l',))):
        tunes = {tuple(load_deployment(model).generator_pythia_settings) for model in models}
        if len(tunes) != 1:
            raise ValueError(f'the {portal} deployments do not share one audited tune')
        settings = out / f'audit-settings-{portal}.txt'
        settings.write_text('\n'.join(tunes.pop()) + '\n')
        support += run([out / 'audit-support', settings, portal], text=True, capture_output=True,
                       env=dict(os.environ, PYTHIA8DATA=config['xmldoc'])).stdout
    spec = importlib.util.spec_from_file_location('replay', ROOT / 'tests/validate_conditional_rng.py')
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    record = dict(schema='exhad-accelerator-build', binary=str(binary), pythia_prefix=str(prefix),
                  commands=commands, pythia_pins_verified=pins,
                  support_audit=support,
                  rng_initializer='exhad_rng_simd; bit-exact Pythia 8.317 state; scalar fallback',
                  rng_state_checks=rng_checks,
                  rng_full_event_replay=replay.validate(config, binary, scalar_binary),
                  domain='Central-generator alp-fermion, m > 2.4 GeV (with the 4-5 GeV transition mixture), '
                         'rate-first eta-pipi family; central-generator scalar portals, m >= 3 GeV, '
                         'rate-first four-pion family; B-L vector current, '
                         '2.0 <= m <= 5.0 GeV, every variation; unit weights; all other public routes '
                         'use the ordinary implementation.')
    (out / 'build.json').write_text(json.dumps(record, indent=2) + '\n')
    print(support, end='')
    return binary


def configure():
    build = dict(line.split('=', 1) for line in
                 (ROOT / 'cpp/.pythia8-build').read_text().splitlines() if '=' in line)
    runtime = ROOT / '.runtime'
    runtime.mkdir(exist_ok=True)
    current = runtime / 'current.json'
    compiler = [*shlex.split(os.environ.get('CXX', 'c++')), '-std=c++17', '-O2']
    includes = ['-I' + str(Path(build['prefix']) / 'include'), '-I' + str(ROOT / 'cpp/include')]
    link_pythia = ['-L' + build['libdir'], '-Wl,-rpath,' + build['libdir'], '-lpythia8']
    subprocess.run([*compiler, str(ROOT / 'tools/decay_worker.cc'), includes[0], *link_pythia,
                    '-o', str(runtime / 'exhad-decay')], check=True)
    subprocess.run([*compiler, str(ROOT / 'tools/shower_worker.cc'),
        str(ROOT / 'cpp/src/symmetry_filter.o'), str(ROOT / 'cpp/src/isospin_cg.o'),
        *includes, *link_pythia, '-o', str(runtime / 'exhad-shower')], check=True)
    config = {'xmldoc': build['xmldoc']}
    write(current, config)  # the gated build below generates events through this configuration
    accelerator = runtime / 'accelerator'
    shutil.rmtree(accelerator, ignore_errors=True)
    accelerator.mkdir()
    started = time.monotonic()

    try:
        binary = build_accelerator(accelerator, compiler, includes, link_pythia,
                                   Path(build['prefix']).resolve(), config)
    except BaseException:
        current.unlink(missing_ok=True)  # an ungated tree is not a configured tree
        raise
    config['conditional_acceleration'] = str(binary.relative_to(ROOT))
    write(current, config)
    print(f'Accelerator gates passed in {time.monotonic() - started:.1f} s: {binary}')
    from exhad.models import MODELS
    print('Configured models: ' + ', '.join(MODELS))


def write(path, config):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(config, indent=2) + '\n')
    temporary.replace(path)


if __name__ == '__main__':
    configure()
