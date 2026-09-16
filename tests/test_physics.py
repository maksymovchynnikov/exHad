"""Live physics regression suite; needs a built and configured checkout (make, tools/configure.py).

python -m unittest tests.test_physics            run (skipped without cpp/exhad and .runtime)
python tests/test_physics.py --update-golden     rewrite the digests in tests/golden_digests.json

Golden digests are sha256 of canonical JSON (sorted keys, repr floats) of the public-API
output of each scenario; a rejected request is stored as {"error": exception class}.
They are written by --update-golden. Same-seed output is deterministic, so any digest change
is a change of physics, random streams or seeds. Both alp-fermion routes are pinned: unweighted
execution="auto" requests above 2.4 GeV (alp-fermion-had-*, alp-fermion-seed-max, variations) run the
accelerator, which tools/configure.py builds, and alp-fermion-ref-* run execution="reference".
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exhad import Generator  # noqa: E402
from exhad.core.pdg import antiparticle, m0  # noqa: E402

CONFIGURED = (ROOT / 'cpp/exhad').is_file() and (ROOT / '.runtime/current.json').is_file()
GOLDEN = Path(__file__).with_name('golden_digests.json')


def primary(mass, w, pdgs):
    """Flat 24-field lepton + massless q-qbar row at pair mass w (the parent at rest)."""
    ml = m0(pdgs[0])
    e = (mass * mass + ml * ml - w * w) / (2 * mass)
    p = math.sqrt(e * e - ml * ml)
    row = [0., 0., p, e, ml, pdgs[0], 0., 1.]
    for pid, sign in zip(pdgs[1:], (1, -1)):
        row += [sign * w / 2, 0., -p / 2, (mass - e) / 2, 0., pid, 0., 0.]
    return row


def check_events(test, events, mass, count):
    """Finite six-field rows, E > 0, on shell to 2e-6 m^2, four-momentum closure to 2e-6 m."""
    test.assertEqual(len(events), count)
    for event in events:
        test.assertTrue(event)
        rows = np.asarray(event, dtype=float)
        test.assertEqual(rows.shape[1], 6)
        test.assertTrue(np.all(np.isfinite(rows)) and np.all(rows[:, 3] > 0) and np.all(rows[:, 4] >= 0))
        test.assertTrue(np.all(rows[:, 5] == np.round(rows[:, 5])))
        shell = rows[:, 3]**2 - (rows[:, :3]**2).sum(axis=1) - rows[:, 4]**2
        test.assertLess(np.abs(shell).max(), 2e-6 * mass**2)
        total = [math.fsum(rows[:, k]) for k in range(4)]
        test.assertLess(max(abs(t - (mass if k == 3 else 0.)) for k, t in enumerate(total)), 2e-6 * mass)


def digest(value):
    text = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def golden_run(scenarios):
    """sha256 per scenario; scenarios sharing model/variation/execution/parallel share one generator."""
    groups = {}
    for sid, s in scenarios.items():
        key = (s['model'], s.get('variation', 'central'), s.get('execution', 'auto'),
               s.get('workers'), s.get('chunk_size'))
        groups.setdefault(key, []).append((sid, s))

    def run(key):
        model, variation, execution, workers, chunk_size = key
        generator = Generator(model, variation=variation, execution=execution, workers=workers,
                              chunk_size=chunk_size or 512)
        result = {}
        with generator:
            for sid, s in groups[key]:
                kw = dict(s.get('kw', {}))
                try:
                    if s['method'] == 'hadronize_hnl':
                        rows = [primary(s['mass'], s['w'], s['pdgs'])] * s['events']
                        value = generator.hadronize_hnl(s['mass'], s['pdgs'], rows, seed=s['seed'])
                    else:
                        value = getattr(generator, s['method'])(s['mass'], s['events'], seed=s['seed'], **kw)
                except (RuntimeError, ValueError) as exc:
                    value = {'error': type(exc).__name__}
                result[sid] = digest(value)
        return result

    with ThreadPoolExecutor(max_workers=4) as pool:
        return {sid: h for part in pool.map(run, groups) for sid, h in part.items()}


def build_record():
    return dict(line.split('=', 1) for line in (ROOT / 'cpp/.pythia8-build').read_text().splitlines() if '=' in line)


def compile_cpp(sources, output, *flags):
    subprocess.run(['c++', '-std=c++17', '-O2', *map(str, sources), *flags, '-o', str(output)], check=True)


@unittest.skipUnless(CONFIGURED, 'build exHad (make -C cpp) and run python tools/configure.py')
class LiveModels(unittest.TestCase):
    def test_golden_digests(self):
        scenarios = json.loads(GOLDEN.read_text())['scenarios']
        actual = golden_run(scenarios)
        changed = sorted(sid for sid, s in scenarios.items() if actual[sid] != s['sha256'])
        self.assertEqual(changed, [], 'same-seed output changed')

    def test_mass_snapshot_equals_the_live_particle_data(self):
        """data/common/pythia8317_m0.json is what tools/snapshot_pythia_masses.py writes from the configured Pythia."""
        spec = importlib.util.spec_from_file_location('snapshot_pythia_masses', ROOT / 'tools/snapshot_pythia_masses.py')
        tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tool)
        xmldoc = json.loads((ROOT / '.runtime/current.json').read_text())['xmldoc']
        self.assertEqual(tool.text(tool.snapshot(xmldoc)), tool.TARGET.read_text())

    def test_accelerator_is_built_and_gated(self):
        config = json.loads((ROOT / '.runtime/current.json').read_text())
        self.assertIn('conditional_acceleration', config, 'tools/configure.py did not record the accelerator')
        binary = ROOT / config['conditional_acceleration']
        self.assertTrue(os.access(binary, os.X_OK), f'accelerator missing: {binary}')
        record = json.loads((binary.parent / 'build.json').read_text())
        self.assertTrue(record['rng_full_event_replay']['full_daughter_records_identical'])
        self.assertEqual(sum(n.startswith('include/') for n in record['pythia_pins_verified']), 4)
        scenarios = json.loads(GOLDEN.read_text())['scenarios']  # the two routes have distinct golden events
        self.assertNotEqual(scenarios['alp-fermion-had-3.000']['sha256'], scenarios['alp-fermion-ref-3.000']['sha256'])

    def test_closure_replay_and_history_independence(self):
        cases = {'dark-photon': (1.9, 4.2), 'alp-fermion': (2.0, 3.0, 4.5), 'scalar-central': (2.5,),
                 'scalar-lower': (3.5,), 'scalar-upper': (4.0,), 'scalar-1809': (2.2,), 'b-l': (2.5, 4.0)}

        def run(model):
            with Generator(model) as generator:
                out = []
                for mass in cases[model]:
                    first = generator.generate(mass, 12, seed=20260914)
                    generator.generate(mass + .1, 3, seed=9)  # an intervening request at another mass
                    out.append((mass, first, generator.generate(mass, 12, seed=20260914)))
                return out
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = dict(zip(cases, pool.map(run, cases)))
        for model, rows in results.items():
            for mass, first, replay in rows:
                with self.subTest(model=model, mass=mass):
                    check_events(self, first, mass, 12)
                    self.assertEqual(first, replay)
        with Generator('hnl') as generator:
            for mass, mixing in ((1.5, [1, 1, 1]), (6., [0, 1, 0])):
                sample = generator.generate_all(mass, 16, seed=5, mixing=mixing)
                check_events(self, sample['events'], mass, 16)
                self.assertEqual(len(sample['channel_labels']), 16)
                self.assertEqual(sample, generator.generate_all(mass, 16, seed=5, mixing=mixing))
            for mass in (.4925, .5975, 1.0175, 3.6425):  # between the last node below an exclusive threshold and that threshold
                check_events(self, generator.generate_all(mass, 16, seed=5, mixing=[1, 2, 3])['events'], mass, 16)
        with Generator('alp-fermion') as generator:  # 3.5535 GeV is below 2 m_tau: the tau+tau- row stays closed
            check_events(self, generator.generate_all(3.5535, 16, seed=5)['events'], 3.5535, 16)
        with Generator('scalar-central') as generator:  # b-bbar has a supplied rate below 2 m(B+) = 10.5585 GeV
            for mass in (10.55, 10.5585):
                check_events(self, generator.generate(mass, 8, seed=5), mass, 8)
                check_events(self, generator.generate_all(mass, 8, seed=5)['events'], mass, 8)

    def test_parallel_worker_count_invariance_and_raw_weights(self):
        with Generator('alp-fermion', workers=1, chunk_size=5) as one, \
                Generator('alp-fermion', workers=3, chunk_size=5) as three:
            self.assertEqual(one.generate(3., 17, seed=31415), three.generate(3., 17, seed=31415))
            weighted = one.generate_weighted(3., 17, seed=315, weight_floor_fraction=.1)
            self.assertEqual(weighted, three.generate_weighted(3., 17, seed=315, weight_floor_fraction=.1))
        with Generator('alp-fermion', workers=1) as generator:  # raw weights are the concatenated chunk weights
            chunks = [generator.generate_weighted(3., n, seed=Generator._seed(315, 17, i),
                                                  weight_floor_fraction=.1) for i, n in enumerate((5, 5, 5, 2))]
        self.assertEqual(weighted['raw_weights'], [w for chunk in chunks for w in chunk['raw_weights']])
        self.assertEqual(weighted['events'], [e for chunk in chunks for e in chunk['events']])
        with Generator('hnl', workers=1, chunk_size=5) as one, \
                Generator('hnl', workers=3, chunk_size=5) as three:
            for mass, w, pdgs in ((2.4, 2., [11, 2, -1]), (2.4, 2., [12, 3, -3]), (3.5, 3., [11, 4, -3])):
                rows = [primary(mass, w, pdgs)] * 11
                events = one.hadronize_hnl(mass, pdgs, rows, seed=88)
                self.assertEqual(events, three.hadronize_hnl(mass, pdgs, rows, seed=88))
                self.assertTrue(all(event[0] == rows[0][:6] for event in events))

    def test_hnl_spectator_exact_w_and_cp_mirror(self):
        cases = (([11, 2, -1], (.5, 1.2, 1.6505, 1.651, 3., 4.5)), ([13, 2, -3], (.7, 2.)),
                 ([11, 4, -1], (2.1, 3.5)), ([11, 4, -3], (3.,)), ([12, 2, -2], (.5, 2., 3.001, 5.)),
                 ([14, 3, -3], (1.1, 3.2, 4.)), ([12, 4, -4], (4.,)), ([12, 5, -5], (12.,)),
                 ([11, 2, -5], (6.,)), ([11, 4, -5], (8.,)))
        with Generator('hnl') as generator:
            for pdgs, ws in cases:
                for w in ws:
                    with self.subTest(pdgs=pdgs, w=w):
                        mass = w + m0(pdgs[0]) + .25
                        row = primary(mass, w, pdgs)
                        events = generator.hadronize_hnl(mass, pdgs, [row] * 4, seed=731)
                        check_events(self, events, mass, 4)
                        for event in events:
                            self.assertEqual(event[0], row[:6])
                            hadrons = np.asarray(event)[1:, :4].sum(axis=0)
                            self.assertLess(abs(hadrons[3]**2 - (hadrons[:3]**2).sum() - w * w), 2e-6)
                        if abs(pdgs[0]) in (11, 13, 15) and w <= 4.:
                            conjugate = [-p for p in pdgs]
                            anti = generator.hadronize_hnl(mass, conjugate, [primary(mass, w, conjugate)] * 4, seed=731)
                            for event, mirror in zip(events, anti):
                                np.testing.assert_allclose(np.asarray(event)[:, :5], np.asarray(mirror)[:, :5])
                                self.assertEqual([antiparticle(int(p[5])) for p in event],
                                                 [int(p[5]) for p in mirror])
            for mass, pdgs, rows in ((40.01, [12, 2, -2], [primary(40.01, 39., [12, 2, -2])]),
                                     (3., [11, 2, -3], [[0.] * 24]), (3., [12, 4, -4], [primary(3., 2.5, [12, 4, -4])])):
                with self.assertRaises(RuntimeError):
                    generator.hadronize_hnl(mass, pdgs, rows)
            for seed in (1722738683, 2**64 - 1):  # the CC_us run server takes a bounded seed
                self.assertEqual(len(generator.hadronize_hnl(2.25, [13, 2, -3], [primary(2.25, 2., [13, 2, -3])], seed=seed)), 1)

    def test_weighted_floor_one_is_the_reference_selection(self):
        with Generator('alp-fermion', execution='reference') as reference, Generator('alp-fermion') as generator:
            for mass in (2.2, 3.):
                expected = reference.generate(mass, 30, seed=7988)
                sample = generator.generate_weighted(mass, 30, seed=7988, weight_floor_fraction=1.)
                self.assertEqual(sample['events'], expected)
                weights = list(zip(sample['raw_weights'], sample['normalization_groups']))
                self.assertLessEqual(len({w for w, g in weights if g == 'fragmentation'}), 1)
                self.assertTrue(all(w == 1. for w, g in weights if g == 'external'))

    def test_cli_weighted_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'weighted.json'
            subprocess.run([sys.executable, '-m', 'exhad', '--model', 'alp-fermion', '--mass', '3.0', '--events', '24',
                            '--weighted', '--output', str(path)], check=True, cwd=ROOT, capture_output=True,
                           env=dict(os.environ, PYTHONPATH=str(ROOT)))
            payload = json.loads(path.read_text())
        self.assertEqual(payload['schema'], 'exhad-weighted-rest-frame-events-v1')
        check_events(self, payload['events'], 3., 24)
        weights = [w for w, g in zip(payload['hadronization_weights'], payload['normalization_groups'])
                   if g == 'fragmentation']
        self.assertAlmostEqual(math.fsum(weights) / len(weights), 1., places=12)

    def test_full_decay_closure_and_bl_charm_width_scale(self):
        from exhad import full_decays, registry
        config = json.loads((ROOT / '.runtime/current.json').read_text())
        os.environ.update(EXHAD_RUNTIME_BINARY=str(ROOT / 'cpp/exhad'), EXHAD_PYTHIA8DATA=config['xmldoc'])
        for model, masses in (('dark-photon', (1.7, 2.5, 4.5, 5.)), ('b-l', (2., 3.8, 5.))):
            for mass in masses:
                labels, ids, p = full_decays.boson_rates(model, mass, 'central')
                self.assertAlmostEqual(math.fsum(p), 1., places=14)
                self.assertEqual((labels[0], ids[0]), ('hadronic', None))
                self.assertEqual(len(labels) - 1, 6 if model == 'b-l' else 3)
        charm = registry.dark_photon(None)[1].owner
        for mass in (3.8, 4.2, 5.):
            point = registry.b_l().rate_point(mass)
            em = charm.closure_at(mass).inclusive_charm_width
            self.assertEqual(point.charm_em.inclusive_charm_width, em)
            self.assertEqual(point.charm_width_per_g_b_squared_gev, .25 * em)
        self.assertEqual(registry.b_l().rate_point(3.7).charm_probability, 0.)

    def test_native_charm_filter_and_seed_initializer(self):
        build = build_record()
        pythia = ['-I' + build['prefix'] + '/include', '-L' + build['libdir'], '-Wl,-rpath,' + build['libdir'], '-lpythia8']
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / 'filter'
            compile_cpp([ROOT / 'tests/check_alp_fermion_charm_filter.cc', ROOT / 'cpp/src/symmetry_filter.o',
                         ROOT / 'cpp/src/isospin_cg.o'], binary, '-I' + str(ROOT / 'cpp/include'), *pythia)
            subprocess.run([binary, build['xmldoc']], check=True, capture_output=True)
            binary = Path(tmp) / 'rng'  # the fast seed initializer reproduces Pythia's Rndm state and draws
            compile_cpp([ROOT / 'tests/check_rng_init.cc'], binary, '-I' + str(ROOT / 'cpp/include'), *pythia)
            subprocess.run([binary], check=True, capture_output=True)

    def test_isospin_filter_accepts_with_the_exact_projection_in_every_record_order(self):
        build = build_record()
        pythia = ['-I' + build['prefix'] + '/include', '-L' + build['libdir'], '-Wl,-rpath,' + build['libdir'], '-lpythia8']
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / 'projector'  # 427 exact P(I | state) rows, all distinct record orders
            compile_cpp([ROOT / 'tests/check_isospin_projector.cc', ROOT / 'cpp/src/symmetry_filter.o',
                         ROOT / 'cpp/src/isospin_cg.o'], binary, '-I' + str(ROOT / 'cpp/include'), *pythia)
            result = subprocess.run([binary, build['xmldoc'], ROOT / 'tests/isospin_projector_fixture.tsv'],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout[-2000:])

    def test_cpp_header_client_equals_python(self):
        models = ('alp-fermion', 'scalar-central', 'dark-photon', 'b-l', 'hnl')
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / 'client'
            compile_cpp([ROOT / 'tests/cpp_batch.cc'], binary, '-Wall', '-Wextra', '-Werror', '-I' + str(ROOT / 'include'))

            def run(model):
                raw = subprocess.check_output([str(binary), str(ROOT), sys.executable, model, '2'], text=True)
                lines, events = iter(raw.splitlines()), []
                for line in lines:  # EVENT <n>, then n particle rows
                    events.append([list(map(float, next(lines).split())) for _ in range(int(line.split()[1]))])
                with Generator(model, execution='reference') as generator:
                    expected = generator.generate_all(2., 5, seed=123, mixing=[1, 0, 0] if model == 'hnl' else None)
                return events, expected['events']
            with ThreadPoolExecutor(max_workers=3) as pool:
                for model, (actual, expected) in zip(models, pool.map(run, models)):
                    with self.subTest(model=model):
                        self.assertEqual(actual, expected)
            bad = subprocess.run([str(binary), str(ROOT), sys.executable, 'alp-fermion', '100'],
                                 capture_output=True, text=True)
            self.assertTrue(bad.returncode != 0 and not bad.stdout and 'exHad:' in bad.stderr)

    def test_pythia8_decay_handler_on_default_mothers(self):
        """Pythia 8.317 hands the default mother of five models to exHad and keeps the host's tau0 with zero width."""
        build = build_record()
        pythia = ['-I' + build['prefix'] + '/include', '-L' + build['libdir'], '-Wl,-rpath,' + build['libdir'], '-lpythia8']
        models = ('dark-photon', 'b-l', 'scalar-central', 'alp-fermion', 'hnl')
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / 'mothers'
            compile_cpp([ROOT / 'tests/check_mother_codes.cc'], binary, '-Wall', '-Wextra', '-Werror',
                        '-I' + str(ROOT / 'include'), *pythia)

            def run(model):
                return subprocess.check_output([str(binary), str(ROOT), sys.executable, build['xmldoc'], model], text=True)
            with ThreadPoolExecutor(max_workers=3) as pool:
                results = [line.split() for line in pool.map(run, models)]
        self.assertEqual(results, [[str(code), '20', '1000', '0'] for code in (4900022, 32, 35, 36, 9900012)])

    def test_the_host_adds_no_radiation_to_the_decay_products(self):
        """Pythia 8.317 shows exHad's own final state for every mother code, and radiates only on request."""
        build = build_record()
        pythia = ['-I' + build['prefix'] + '/include', '-L' + build['libdir'], '-Wl,-rpath,' + build['libdir'], '-lpythia8']
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / 'radiation'
            compile_cpp([ROOT / 'tests/check_host_radiation.cc'], binary, '-Wall', '-Wextra', '-Werror',
                        '-I' + str(ROOT / 'include'), *pythia)
            output = subprocess.check_output([str(binary), str(ROOT), sys.executable, build['xmldoc']], text=True)
        rows = [list(map(int, line.split())) for line in output.splitlines()]
        self.assertEqual([row[0] for row in rows], [4900022, 32, 35, 36, 9900012])
        for mother, pairs, shower, extra, photons, pairs_on, photons_on in rows:
            with self.subTest(mother=mother):  # 200 decays of 2 GeV, of which `pairs` gave an e or mu pair
                self.assertGreater(pairs, 0)
                self.assertEqual([shower, extra, photons], [0, 0, 0])
                self.assertEqual(pairs_on, pairs)  # the same decays, with the host allowed to radiate
                self.assertGreater(photons_on, 0)

    def test_header_client_rejects_malformed_replies(self):
        """A fake worker's reply with a non-finite or trailing token, a PDG 211.5 or an empty event fails the batch."""
        row = '0 0 1 1 0 22'
        valid = ['OK 5'] + ['EVENT 1', row] * 5 + ['END']
        variants = [valid] + [valid[:k] + list(lines) + valid[k + drop:] for k, drop, lines in (
            (2, 1, ['0 0 nan 1 0 22']), (2, 1, ['0 0 1 inf 0 22']), (2, 1, [row + ' 7']), (2, 1, ['0 0 1 1 0 211.5']),
            (1, 2, ['EVENT 0']), (1, 1, ['EVENT 1 1']), (0, 1, ['OK 5 5']))]
        with tempfile.TemporaryDirectory() as tmp:
            binary, fake = Path(tmp) / 'client', Path(tmp) / 'exhad'
            compile_cpp([ROOT / 'tests/cpp_batch.cc'], binary, '-I' + str(ROOT / 'include'))
            fake.mkdir()
            (fake / 'worker.py').write_text('import sys\nreply = open(sys.argv[0][:-9] + "reply.txt").read()\n'
                                            'for line in sys.stdin:\n    sys.stdout.write(reply)\n    sys.stdout.flush()\n')
            for index, lines in enumerate(variants):
                (fake / 'reply.txt').write_text('\n'.join(lines) + '\n')
                run = subprocess.run([str(binary), tmp, sys.executable, 'alp-fermion', '2'], capture_output=True, text=True)
                with self.subTest(reply=lines):
                    if index == 0:
                        self.assertEqual((run.returncode, run.stdout.count('EVENT 1\n')), (0, 5))
                    else:
                        self.assertTrue(run.returncode == 1 and not run.stdout and 'exHad:' in run.stderr)

    def test_worker_rejects_malformed_numbers_and_run_tables(self):
        env = dict(os.environ, PYTHIA8DATA=build_record()['xmldoc'])

        def run(args, stdin=''):
            result = subprocess.run([str(ROOT / 'cpp/exhad'), *args], input=stdin, capture_output=True, text=True, env=env)
            return result.returncode, result.stdout.splitlines()
        base = ['2', '-1', '-1', '0', 'qq', '--parentSpin=carried', '--set=HadronLevel:mStringMin=1.0']
        invalid = [['--component=rho', option] for option in ('--stopMass=nan', '--stopMass=-1')]
        for options in (['--component=rho', '--stopMass=0'], ['--inject=211,-211'], *invalid, ['--inject=211.5,-211']):
            code, lines = run([*base, *options], 'QUIT\n')
            valid = options[-1] in ('--stopMass=0', '--inject=211,-211')
            self.assertEqual((code, lines[-1][:6], len(lines) > 1), (0, 'BYE', True) if valid else (2, 'Fatal:', False), options)
        for mass in ('nan', 'inf', '-1', '0', '2abc'):  # the mass travels with every request
            code, lines = run([*base, '--component=rho'], f'REQUEST 0 1701 {mass}\nQUIT\n')
            self.assertEqual((code, lines[0][:5], lines[-1][:6], len(lines)), (2, 'READY', 'Fatal:', 2), mass)
        request = 'RUN 2 -1 none -1 averaged 4 7 {mass} {weights} {sets} {modes} {morph}\n{rows}\n'
        good = dict(mass='1.2', weights='1,1', sets=0, modes=1, morph=0, rows='0.999999 -211 111')
        for change in (dict(mass='nan'), dict(mass='1.2x'), dict(weights='nan,1'), dict(weights='-1,1'),
                       dict(rows='nan -211 111'), dict(rows='-0.5 -211 111'), dict(rows='x -211 111'),
                       dict(rows='0.5 -211.5 111'), dict(rows='0.5 -211'), dict(modes=0, morph=1, rows='1 -211'),
                       dict(sets=1, rows='HadronLevel:mStringMin=1.0\n0.999999 -211 111')):
            code, lines = run(['--runs'], request.format(**dict(good, **change)))
            self.assertTrue(code == 2 and lines == [lines[-1]] and lines[-1].startswith('Fatal'), change)
        # A valid run, then (same server and decayers) zero-sum weights, an unknown daughter ID or a
        # daughter whose decays reach a partonic channel (they would read the string settings).
        for change in (dict(weights='0,0'), dict(rows='0.5 -211 7777777'), dict(rows='0.5 443 111')):
            code, lines = run(['--runs'], request.format(**good) + request.format(**dict(good, **change)))
            self.assertEqual((code, lines.count('E'), lines[-2], lines[-1][:6]), (2, 4, 'END 4', 'Fatal:'))

    def test_workers_carry_no_mass_or_settings_history(self):
        env = dict(os.environ, PYTHIA8DATA=build_record()['xmldoc'])

        def run(args, stdin):
            return subprocess.run([str(ROOT / 'cpp/exhad'), *args], input=stdin + 'QUIT\n', capture_output=True,
                                  text=True, env=env, check=True).stdout
        # A worker serves every mass: a request answers as in a fresh worker, whatever came before.
        worker = ['2', '-1', 'none', '-1', 'qq', '--component=ccvec', '--parentSpin=averaged',
                  '--set=HadronLevel:mStringMin=1.0', '--stopMass=0']
        ready, first, second, third = run(worker, 'REQUEST 0 1701 2.0\nREQUEST 1 1702 3.1\nREQUEST 0 1701 2.0\n').split('RESULT ')
        self.assertEqual((third, run(worker, 'REQUEST 1 1702 3.1\n')), (first + 'BYE\n', ready + 'RESULT ' + second + 'BYE\n'))
        # The run server shares its decayers across string settings: a run answers as in a fresh server.
        tuned = ('RUN 2 -1 none -1 averaged 20 7 1.6 1,1 2 1 0\n'
                 'StringZ:aLund=0.9\nStringFlav:probStoUD=0.3\n0.3 -211 111\n')
        plain = 'RUN 2 -1 none -1 averaged 20 9 1.2 1,1 0 1 0\n0.3 -211 111\n'
        alone = run(['--runs'], plain)
        self.assertTrue(alone.endswith('END 20\n') and run(['--runs'], tuned + plain).endswith('END 20\n' + alone))

    def test_batch_kernel_rejects_nonfinite_source_weight(self):
        config = json.loads((ROOT / '.runtime/current.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'settings.txt').write_text('')
            run = subprocess.run([str(ROOT / config['conditional_acceleration']), '3.0', str(Path(tmp) / 'settings.txt'), 'alp-fermion'],
                                 input='CONFIG 2\npseudoscalar-gluon nan 1\n', capture_output=True, text=True,
                                 env=dict(os.environ, PYTHIA8DATA=config['xmldoc']))
        self.assertEqual((run.returncode, run.stderr.splitlines()[-1]), (1, 'invalid source filter'))

    def test_scalar_batch_kernel_draws_four_pions(self):
        config = json.loads((ROOT / '.runtime/current.json').read_text())
        from exhad.accelerator.sampler import _Batch
        from exhad.model1 import sampler
        from exhad.model1.families import classify_channel
        deployment = sampler.load_deployment('scalar-central')
        point = deployment.family_model.evaluate(3.6)
        batch = _Batch(ROOT / config['conditional_acceleration'], deployment, point, {'four-pion'}, 'scalar')
        try:
            events = batch.draw([f'{9100 + i} {i % 2}' for i in range(40)])
        finally:
            batch.close()
        for i, (particles, source, channel) in enumerate(events):
            family = classify_channel('scalar-families', sampler._parse_removed_key(channel))
            self.assertEqual(family == 'four-pion', i % 2 == 1)
            if i % 2:
                self.assertEqual((source, len(particles)), ('glue', 4))
            self.assertAlmostEqual(sum(p[3] for p in particles), 3.6, delta=1e-5)

    def test_batch_kernel_rejects_an_unknown_portal(self):
        config = json.loads((ROOT / '.runtime/current.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'settings.txt').write_text('')
            run = subprocess.run([str(ROOT / config['conditional_acceleration']), '3.0',
                                  str(Path(tmp) / 'settings.txt'), 'dark-photon'],
                                 input='', capture_output=True, text=True,
                                 env=dict(os.environ, PYTHIA8DATA=config['xmldoc']))
        self.assertEqual((run.returncode, run.stderr.splitlines()[-1]),
                         (1, 'unsupported conditional portal dark-photon'))

    def test_baryon_guard_is_the_pair_threshold_and_ownership_decides_above(self):
        build = build_record()
        pythia = ['-I' + build['prefix'] + '/include', '-L' + build['libdir'], '-Wl,-rpath,' + build['libdir'], '-lpythia8']
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / 'guard'  # synthetic final states against 2 m_p
            compile_cpp([ROOT / 'tests/check_baryon_guard.cc', ROOT / 'cpp/src/symmetry_filter.o', ROOT / 'cpp/src/isospin_cg.o'],
                        binary, '-I' + str(ROOT / 'cpp/include'), '-pthread', *pythia)
            subprocess.run([binary, build['xmldoc']], check=True, capture_output=True)
        # Just above 2 m_p, at 1.879 GeV (enhanced diquarks), proton pairs are generated, or vetoed when owned.
        worker = [str(ROOT / 'cpp/exhad'), '2', '-1', '-1', '0', 'qq', '--component=rho',
                  '--parentSpin=carried', '--set=HadronLevel:mStringMin=1.0', '--set=StringFlav:probQQtoQ=1.0']
        requests = ''.join(f'REQUEST {i} {9700400 + i} 1.879\n' for i in range(300)) + 'QUIT\n'
        for veto in ((), ('--veto=-2212:1 2212:1',)):
            result = subprocess.run([*worker, *veto], input=requests, capture_output=True, text=True,
                                    env=dict(os.environ, PYTHIA8DATA=build['xmldoc']))
            lines, finals = result.stdout.splitlines(), []
            for line in lines:
                tokens = line.split()
                if tokens[0] == 'E':
                    finals.append([])
                elif tokens[0] == 'N' and tokens[8] == '1' and tokens[2] != '0':
                    finals[-1].append(int(tokens[2]))
            pairs = sum(sorted(ids) == [-2212, 2212] for ids in finals)
            self.assertEqual((result.returncode, lines[-1], sum(line.startswith('RESULT') for line in lines)), (0, 'BYE', 300))
            self.assertTrue(pairs == 0 if veto else pairs > 0, (veto, pairs))

    def test_hnl_charm_primaries_at_the_router_threshold(self):
        from exhad.hnl import THRESHOLDS
        charge = {211: 1, 321: 1, 2212: 1, 11: -1, 13: -1, 22: 0, 130: 0, 2112: 0, 12: 0, 14: 0, 16: 0}
        with Generator('hnl') as generator:
            for pdgs, current in (([11, 4, -1], 'CC_cd'), ([11, 4, -3], 'CC_cs')):
                for w in (THRESHOLDS[current] + 1e-4, THRESHOLDS[current] + 4.7e-3):
                    mass = w + m0(11) + .25
                    row = primary(mass, w, pdgs)
                    events = generator.hadronize_hnl(mass, pdgs, [row] * 16, seed=9000101)
                    check_events(self, events, mass, 16)
                    for event in events:
                        hadrons = np.asarray(event[1:])
                        self.assertEqual(event[0], row[:6])
                        np.testing.assert_allclose(hadrons[:, :4].sum(axis=0), np.add(row[8:12], row[16:20]), rtol=0, atol=2e-9 * w)
                        self.assertEqual(sum(charge[abs(int(p))] * np.sign(p) for p in hadrons[:, 5]), 1)

    def test_helper_workers_conserve_charge_and_baryon_number(self):
        """Partonic decay channels (charm and bottom baryons, B+) fragment at full hadron level in both workers."""
        from exhad.full_decays import primary
        from exhad.secondary import decayer
        from exhad.models import CHARGED_CURRENT_SOURCE, PSEUDOSCALAR_SOURCE
        from exhad.transition import finish_partons
        config = json.loads((ROOT / '.runtime/current.json').read_text())
        charge = {11: -1, 13: -1, 211: 1, 321: 1, 2212: 1, 12: 0, 14: 0, 16: 0, 22: 0, 130: 0, 2112: 0}  # partons: KeyError

        def totals(event):
            ids = [int(p[5]) for p in event]
            return sum(np.sign(i) * charge[abs(i)] for i in ids), sum(np.sign(i) * (abs(i) in (2212, 2112)) for i in ids)
        # Pythia 8.317 m0; (Q, B) of the particle. moreDecays() would violate Q or B in 44-100% of these decays.
        for pid, m, qb in ((4122, 2.28646, (1, 1)), (4132, 2.47088, (0, 1)), (4332, 2.6952, (0, 1)),
                           (521, 5.27925, (1, 0)), (5122, 5.6194, (0, 1))):
            for sign in (1, -1):
                events = decayer(config['xmldoc']).finish([[[0., 0., 0., m, m, sign * pid]]] * 200, pid + sign)
                with self.subTest(pdg=sign * pid):
                    self.assertEqual({totals(event) for event in events}, {(sign * qb[0], sign * qb[1])})
        for mode, mass, ids in ((PSEUDOSCALAR_SOURCE, 4.8, [4, -4]), (CHARGED_CURRENT_SOURCE, 12., [5, -5])):
            rows = primary(mass, ids, 100, 17).reshape(100, 2, 8)[:, :, :6].tolist()
            events = finish_partons(config, rows, 17, mode)
            with self.subTest(mode=mode):
                check_events(self, events, mass, 100)
                self.assertEqual({totals(event) for event in events}, {(0, 0)})

    def test_generate_rows_conserve_and_replay(self):
        """Every open explicit table row: closure, charge and baryon number, both terminals, label-independent seeds."""
        from exhad.core.kinematics import FICTITIOUS
        from exhad.models import model_info
        charge = {11: -1, 13: -1, 15: -1, 211: 1, 321: 1, 2212: 1, 12: 0, 14: 0, 16: 0, 22: 0, 111: 0, 130: 0, 310: 0, 2112: 0}
        for model, masses, mixing in (('dark-photon', (.5, 1.02, 1.69), None), ('alp-fermion', (.6, 1.5), None),
                                      ('scalar-central', (1.2, 1.99), None), ('b-l', (1.9,), None),
                                      ('hnl', (.5, 1.8, 3.), [1, 2, 3])):
            table = json.loads(Path(model_info(model)['tables']['decay']).read_text())
            with Generator(model) as generator:
                for mass in masses:
                    rows = {row[0]: 6 for row in table if not any(abs(int(p)) in FICTITIOUS for p in row[1])
                            and sum(m0(int(p)) for p in row[1] if int(p) != -999) < mass}
                    if model == 'hnl':  # a row without width at this mass and these mixings raises
                        from exhad.full_decays import hnl_rates
                        labels, _, _, p = hnl_rates(mass, mixing)
                        rows = {label: n for label, n in rows.items()
                                if sum(q for name, q in zip(labels, p) if name == label) > 0}
                    for terminal in ('pythia', 'matched'):
                        result = generator.generate_rows(mass, rows, seed=41, terminal=terminal, mixing=mixing)
                        with self.subTest(model=model, mass=mass, terminal=terminal):
                            self.assertEqual(list(result), list(rows))
                            for label, events in result.items():
                                check_events(self, events, mass, 6)
                                signs = [(int(np.sign(p[5])), abs(int(p[5]))) for event in events for p in event]
                                self.assertTrue(terminal == 'pythia' or not {111, 310} & {pid for _, pid in signs})
                                for event in events:
                                    ids = [int(p[5]) for p in event]
                                    self.assertEqual(sum(np.sign(i) * charge[abs(i)] for i in ids), 0, label)
                                    self.assertEqual(sum(np.sign(i) * (abs(i) in (2212, 2112)) for i in ids), 0, label)
                    label = next(iter(rows))
                    self.assertEqual(generator.generate_rows(mass, {label: 6}, seed=41, terminal='matched', mixing=mixing)[label],
                                     result[label])
        with Generator('dark-photon') as generator:
            for rows, message in (({'Jets-uu': 1}, 'partonic'), ({'ppbar': 1}, 'closed'), ({'nope': 1}, 'Unknown')):
                with self.assertRaisesRegex(RuntimeError, message):
                    generator.generate_rows(1.5, rows)
        with Generator('b-l') as generator:  # B-L is isoscalar: two pions are not one of its rows
            with self.assertRaisesRegex(RuntimeError, 'Unknown b-l decay-table row'):
                generator.generate_rows(1.5, {'Pip_Pim': 1})

    def test_generate_rows_tau_rows_decay_fully_under_both_terminals(self):
        """Rows whose only unstable particles are taus are identical under 'matched' and 'pythia'."""
        for model, mass, rows, mixing in (('dark-photon', 4., {'tau-pair': 20}, None),
                                          ('hnl', 2., {'Pitau': 20, 'etauv': 20}, [0, 0, 1])):
            with Generator(model) as generator:
                matched = generator.generate_rows(mass, rows, seed=13, terminal='matched', mixing=mixing)
                self.assertEqual(matched, generator.generate_rows(mass, rows, seed=13, terminal='pythia', mixing=mixing))
                for label, events in matched.items():
                    check_events(self, events, mass, rows[label])
        with Generator('hnl') as generator:  # the pi0 takes the matched decay, the tau Pythia's
            events = generator.generate_rows(2.3, {'2Pitau': 20}, seed=13, terminal='matched', mixing=[0, 0, 1])['2Pitau']
        check_events(self, events, 2.3, 20)
        self.assertFalse({111, 310} & {abs(int(p[5])) for event in events for p in event})

    def test_generate_rows_parallel_and_header_client(self):
        rows = {'Pip_Pim_2Pi0': 7, 'EtaOmega': 3, 'KL_KS': 0}
        with Generator('dark-photon', workers=1, chunk_size=4) as one, \
                Generator('dark-photon', workers=3, chunk_size=4) as three:
            expected = one.generate_rows(1.6, rows, seed=5, terminal='matched')
            self.assertEqual(three.generate_rows(1.6, rows, seed=5, terminal='matched'), expected)
            self.assertEqual(three.generate_rows(1.6, {}, seed=5, terminal='matched'), {})  # validated, no events
            self.assertEqual(three.generate_rows(1.6, {'EtaOmega': 3}, seed=5, terminal='matched')['EtaOmega'], expected['EtaOmega'])
        source = """#include "exhad/Generator.hpp"
#include <iostream>
int main(int, char** argv) {
  exhad::Generator generator(argv[1], argv[2], "hnl");
  if (!generator.generateRows(1.2, {}, 9, "matched", {1., 2., 3.}).empty()) return 1;  // as in Python
  for (const auto& events : generator.generateRows(1.2, {{"2Piv", 3}, {"Pie", 2}}, 9, "matched", {1., 2., 3.}))
    for (const auto& event : events) {
      std::cout << "EVENT " << event.size() << '\\n';
      for (const auto& p : event)
        std::cout << std::setprecision(17) << p.px << ' ' << p.py << ' ' << p.pz << ' ' << p.energy << ' ' << p.mass << ' ' << p.pdg << '\\n';
    }
}
"""
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'rows.cc').write_text(source)
            compile_cpp([Path(tmp) / 'rows.cc'], Path(tmp) / 'rows', '-Wall', '-Wextra', '-Werror', '-I' + str(ROOT / 'include'))
            lines, events = iter(subprocess.check_output([str(Path(tmp) / 'rows'), str(ROOT), sys.executable], text=True).splitlines()), []
            for line in lines:
                events.append([list(map(float, next(lines).split())) for _ in range(int(line.split()[1]))])
        with Generator('hnl') as generator:
            expected = generator.generate_rows(1.2, {'2Piv': 3, 'Pie': 2}, seed=9, terminal='matched', mixing=[1., 2., 3.])
        self.assertEqual(events, expected['2Piv'] + expected['Pie'])

    def test_starved_dedicated_hnl_runs_continue_without_bias(self):
        """Forced light-NC categories that exhaust a run's attempt budget are completed by fresh-seed runs."""
        from scipy import stats
        from exhad.hnl_backends import dedicated as d
        self.assertEqual(len(d._nc_light(1.15, 1, 179)), 1)  # a count-1 request whose first run is starved is completed
        events = d._nc_light(1.025, 32, 5)
        self.assertEqual(len({tuple(event) for event in events}), 32)  # nothing recycled
        three = d._forced('HNL_NC', {'3Pi'})  # a request that is not short is one unchanged run
        self.assertEqual(d._complete(lambda k, s: d._string('HNL_NC', 1.15, k, s, three), 4, 11, ''),
                         d._string('HNL_NC', 1.15, 4, 11, three))
        closed = d._nc(0.275, 8, 5, 'light')  # the 0.275-GeV bin lies below pi+ pi-: generated at 0.2801 GeV
        self.assertEqual({tuple(sorted(p[0] for p in event)) for event in closed}, {(-211, 211)})
        for event in closed:
            self.assertAlmostEqual(sum(p[4] for p in event), .2801, places=9)
        other, arms = d._forced('HNL_NC', {'5Pi', 'other'}), ([], [])
        for r in range(400):  # count-1 requests at 1.025 GeV starve their first run about half the time
            sizes = []

            def run(k, s):
                sizes.append(len(got := d._string('HNL_NC', 1.025, k, s, other, mass='%.17g')))
                return got
            (event,) = d._complete(run, 1, 9_400_000 + r, '')
            arms[sizes[0] == 0].append(event)  # first-run events, continuation-run events
        self.assertGreater(min(map(len, arms)), 150)
        charged = [[sum(abs(p[0]) in (11, 13, 211, 321, 2212) for p in e) for e in arm] for arm in arms]
        leading = [[max(math.sqrt(p[1]**2 + p[2]**2 + p[3]**2) for p in e) for e in arm] for arm in arms]
        self.assertGreaterEqual(stats.ks_2samp(*charged).pvalue, 1e-3)
        self.assertGreaterEqual(stats.ks_2samp(*leading).pvalue, 1e-3)
        keys = [[' '.join(map(str, sorted(p[0] for p in e))) for e in arm] for arm in arms]
        common = sorted({k for k in keys[0] + keys[1] if (keys[0] + keys[1]).count(k) >= 20})
        table = [[arm.count(k) for k in common] + [sum(k not in common for k in arm)] for arm in keys]
        self.assertGreaterEqual(stats.chi2_contingency(np.asarray(table)[:, np.sum(table, axis=0) > 0]).pvalue, 1e-3)


if __name__ == '__main__':
    if sys.argv[1:] == ['--update-golden']:
        document = json.loads(GOLDEN.read_text())
        for sid, value in golden_run(document['scenarios']).items():
            document['scenarios'][sid]['sha256'] = value
        rows = ',\n'.join(f'  {json.dumps(sid)}: {json.dumps(s)}' for sid, s in document['scenarios'].items())
        GOLDEN.write_text(f'{{"note": {json.dumps(document["note"])},\n "scenarios": {{\n{rows}\n}}}}\n')
    else:
        unittest.main()
