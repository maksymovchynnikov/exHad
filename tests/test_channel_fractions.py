"""The channel classifier of examples/plot_channel_fractions.py, and a two-point smoke run.

python -m unittest tests.test_channel_fractions   (the smoke run is skipped without a build)
"""
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIGURED = (ROOT / 'cpp/exhad').is_file() and (ROOT / '.runtime/current.json').is_file()


def example():
    spec = importlib.util.spec_from_file_location('plot_channel_fractions',
                                                  ROOT / 'examples/plot_channel_fractions.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PLOT = example()


def particle(pdg, momentum, mass):
    """One event record entry along z with the given three-momentum and mass."""
    return [0., 0., momentum, math.hypot(momentum, mass), mass, pdg]


def back_to_back(pdgs, masses, parent):
    """Two daughters of a parent at rest: their invariant mass is the parent's."""
    p = math.sqrt((parent**2 - (masses[0] + masses[1])**2)
                  * (parent**2 - (masses[0] - masses[1])**2)) / (2 * parent)
    return [particle(pdgs[0], p, masses[0]), particle(pdgs[1], -p, masses[1])]


class Classifier(unittest.TestCase):
    def test_pi0_and_eta_are_rebuilt_from_their_photons(self):
        pi0 = back_to_back((22, 22), (0., 0.), PLOT.m0(111))
        eta = back_to_back((22, 22), (0., 0.), PLOT.m0(221))
        self.assertEqual(PLOT.hadrons(pi0), {111: 1})
        self.assertEqual(PLOT.hadrons(eta), {221: 1})
        # An unpaired photon stays a photon: no parent is invented.
        self.assertEqual(PLOT.hadrons([particle(22, 1., 0.)]), {22: 1})

    def test_k_short_is_rebuilt_from_its_charged_pions(self):
        charged = back_to_back((211, -211), (PLOT.m0(211),) * 2, PLOT.m0(310))
        self.assertEqual(PLOT.hadrons(charged), {310: 1})
        # Two pi0 at rest are two pi0, not the K_S whose mass their pair does not carry.
        photons = [p for _ in range(2) for p in back_to_back((22, 22), (0., 0.), PLOT.m0(111))]
        self.assertEqual(PLOT.hadrons(photons), {111: 2})

    def test_groups_of_counted_hadrons(self):
        from collections import Counter
        cases = {('2pi',): [211, -211], ('3pi',): [211, -211, 111], ('4pi',): [211, -211, 211, -211],
                 ('5pi',): [211, -211, 111, 211, -211], ('6pi',): [211, -211] * 3,
                 ('kk',): [321, -321], ('kkpi',): [321, -321, 111], ('kkpipi',): [310, 130, 211, -211],
                 ('eta',): [221, 211, -211], ('etagamma',): [221, 22], ('pi0gamma',): [111, 22],
                 ('nnbar',): [2212, -2212], ('other',): [211, -211, 13, -13]}
        for (group,), pdgs in cases.items():
            counts = Counter(abs(pdg) if pdg in (-211, -321, -2212, -2112) else pdg for pdg in pdgs)
            self.assertEqual(PLOT.channel_group(counts), group, pdgs)
        # Seven pions and a kaon pair with three pions leave the named groups.
        self.assertEqual(PLOT.channel_group(Counter([211, -211] * 3 + [111])), 'other')
        self.assertEqual(PLOT.channel_group(Counter([321, -321, 211, -211, 111])), 'other')

    def test_mass_points_stay_inside_the_generation_range(self):
        info = PLOT.model_info('dark-photon')
        for nonhadronic in (False, True):
            low, high = info['generation_gev']['all' if nonhadronic else 'hadronic']
            masses = PLOT.masses_of(info, nonhadronic, 7)
            self.assertEqual(len(masses), 7)
            self.assertTrue(low <= masses[0] < masses[-1] <= high, masses)


@unittest.skipUnless(CONFIGURED, 'build exHad (make -C cpp) and run python tools/configure.py')
class SmokeRun(unittest.TestCase):
    def test_two_points_of_one_model_produce_a_figure(self):
        # The lowest of the three points has no open hadronic channel and is left out.
        result = PLOT.scan('dark-photon', 3, 12, 2, 1, False)
        self.assertGreaterEqual(len(result['points']), 2)
        for row in result['points']:
            self.assertEqual(row['events'], 12)
            self.assertAlmostEqual(math.fsum(row['fractions'].values()), 1.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'channel_fractions_dark-photon.pdf'
            PLOT.draw(result, path)
            self.assertTrue(path.is_file() and path.stat().st_size > 0)


if __name__ == '__main__':
    unittest.main()
