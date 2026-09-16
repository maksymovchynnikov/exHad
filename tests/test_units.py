"""Deterministic regression tests that need no Pythia build (run in CI)."""
from collections import Counter
from operator import itemgetter
import io
import json
import math
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exhad import DATA, Generator, hnl, output, worker  # noqa: E402
from exhad.b_l import open_charm as bl  # noqa: E402
from exhad.core.pdg import m0  # noqa: E402
from exhad.dark_photon.charm_rates import build_open_charm_owner  # noqa: E402
from exhad.full_decays import boson_rates, hnl_rates, hnl_tables, normalized, primary  # noqa: E402
from exhad.hnl import channel_spec  # noqa: E402
from exhad.model1 import alp_fermion_open_charm as charm  # noqa: E402
from exhad.model1.charge_matching import ConditionalChannelMatching  # noqa: E402
from exhad.models import check_request, validate_model  # noqa: E402
from exhad.transition import boost, pythia_fraction  # noqa: E402


class FullDecays(unittest.TestCase):
    def test_boson_tables_close_and_scalar_has_no_two_photon_channel(self):
        for model in ('alp-fermion', 'scalar-central', 'scalar-lower', 'scalar-upper', 'scalar-1809'):
            for mass in np.linspace(2., 5., 31):
                labels, ids, p = boson_rates(model, float(mass), 'central')
                self.assertAlmostEqual(math.fsum(p), 1., places=14)
                self.assertTrue(min(p) >= 0. and ids[0] is None)
                self.assertEqual('2gamma' in labels, model == 'alp-fermion')
        for model, high in (('alp-fermion', 5.001), ('scalar-central', 63.001)):
            with self.assertRaises(ValueError):
                boson_rates(model, high, 'central')

    def test_incomplete_tables_are_rejected_not_renormalized(self):
        for rates in ([.3, .5], [-.01, 1.01], [float('nan'), 1.]):
            with self.assertRaises(ValueError):
                normalized(rates)

    def test_hnl_invisible_width_and_mixing(self):
        rows, widths = hnl_tables()
        tabulated = {row[0] for row in rows}
        for mass in (.02, .2, 1., 2., 4., 5.27, 6., 10., 20., 40.):
            for mix in ([1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 2., 3.]):
                labels, ids, _, p = hnl_rates(mass, mix)
                self.assertAlmostEqual(math.fsum(p), 1.)
                # The tabulated total width of the mixing, which no channel of exHad adds to
                gamma = sum(u * np.interp(mass, widths[:, 0], widths[:, i + 1]) for i, u in enumerate(mix))
                # G_F^2 m^5/(96 pi^3) at the grid nodes, interpolated like the other rows
                invisible = (1.166379e-5)**2 * widths[:, 0]**5 / (96 * math.pi**3)
                expected = sum(u * np.interp(mass, widths[:, 0], widths[:, i + 1])
                               * np.interp(mass, widths[:, 0], invisible / widths[:, i + 1])
                               for i, u in enumerate(mix)) / gamma
                self.assertAlmostEqual(sum(w for label, w in zip(labels, p) if label == '3nu'), expected, places=8)
                # Every channel is a row of the tables, or the three neutrinos the rows leave
                self.assertLessEqual(set(labels), tabulated | {'3nu'})
                self.assertTrue(all(isinstance(pid, int) for pdgs in ids for pid in pdgs))
        for mix in (None, [0, 0, 0], [-1, 1, 1], [1, 2], [float('nan'), 0, 1]):
            with self.assertRaises(ValueError):
                hnl_rates(2., mix)
        for mass in (.0199, 40.001):
            with self.assertRaises(ValueError):
                hnl_rates(mass, [1, 0, 0])

    def test_exclusive_rates_vanish_at_their_thresholds(self):
        """Zero at and below the daughter-mass sum, linear from it to the first node above; tabulated nodes are returned exactly.

        Every tabulated node at or below an exclusive threshold is zero, so a closed channel can only be
        positive between the last node below its threshold and the threshold: that whole interval is
        scanned in 0.05-MeV steps for each pure mixing.
        """
        from exhad.core.kinematics import FICTITIOUS
        from exhad.full_decays import interpolate, row_threshold, table
        self.assertEqual([interpolate([[1., 0.], [2., 0.], [3., 4.]], m, 2.5) for m in (2.2, 2.5, 2.75, 3.)],
                         [0., 0., 2., 4.])

        def closed(pdgs, mass):  # a parton pair opens at its lightest mesons
            return sum(FICTITIOUS[abs(p)] if abs(p) in FICTITIOUS else m0(p) for p in pdgs) >= mass

        rows = hnl_tables()[0]
        nodes, grid = rows[0][2][:, 0], set()
        for row in rows:
            pdgs = [int(p) for p in row[1] if int(p) != -999]
            threshold = sum(FICTITIOUS[abs(p)] if abs(p) in FICTITIOUS else m0(p) for p in pdgs)
            if any(abs(p) in FICTITIOUS for p in pdgs):  # e.g. u-bbar e: supplied rate below 5.42011 GeV
                grid.update((threshold - 5e-5, threshold))
                continue
            self.assertFalse(any(row[2 + a][nodes <= threshold, 1].any() for a in range(3)), row[0])
            below = nodes[np.searchsorted(nodes, threshold, 'right') - 1]
            grid.update(np.arange(math.floor(below * 2e4), math.floor(threshold * 2e4) + 2) / 2e4)
        failures = [(mass, alpha, label) for mass in sorted(m for m in grid if m >= .02) for alpha in range(3)
                    for label, pdgs, p in zip(*itemgetter(0, 1, 3)(hnl_rates(mass, np.eye(3)[alpha])))
                    if p > 0 and closed(pdgs, mass)]
        self.assertEqual(failures, [])
        for model in ('alp-fermion', 'scalar-central', 'scalar-lower', 'scalar-upper', 'scalar-1809'):  # tau+tau- opens at 3.55364 GeV
            for mass in np.arange(71060, 71074) / 2e4:
                labels, ids, p = boson_rates(model, float(mass), 'central')
                self.assertFalse(any(v > 0 and closed(i, mass) for i, v in zip(ids[1:], p[1:])), (model, mass))
        bb = next(r[2] for r in table(str(DATA / 'scalar/eventcalc_branching_ratios_2407.13587_central.json'))
                  if r[0] == 'Jets-bb')  # a parton pair opens at two of its lightest hadrons: 2 m(B+)
        self.assertAlmostEqual(row_threshold([5, -5]), 10.5585)
        self.assertGreater(interpolate(bb, 10.55), 0.)
        self.assertEqual([interpolate(bb, m, row_threshold([5, -5])) > 0 for m in (10.55, 10.5585, 10.5586)],
                         [False, False, True])
        rows = table(str(DATA / 'scalar/eventcalc_branching_ratios_2407.13587_central.json'))
        supplied = math.fsum(interpolate(r[2], 10.55) for r in rows if r[0] not in {'ePeM', 'muPmuM', 'tauPtauM'})
        closed_bb = interpolate(bb, 10.55)  # the complete decays renormalize to the open rates
        self.assertAlmostEqual(boson_rates('scalar-central', 10.55, 'central')[2][0],
                               (supplied - closed_bb) / (1 - closed_bb), places=14)

    def test_matched_terminal_keeps_full_tau_decays(self):
        """Taus decay fully in Pythia under both terminals; the matched convention covers the other particles."""
        from exhad.full_decays import realize_rows, subseed
        calls = []

        def decayer(xml, stable=()):
            def finish(events, seed):
                calls.append((stable, [[int(p[5]) for p in event] for event in events], seed))
                return [[[*p[:5], 22.] for p in event] for event in events]
            return SimpleNamespace(finish=finish)
        rows = {'2Pitau': ([211, 111, 15], '1.'), 'Dtau': ([411, 15], '1.'), 'Pitau': ([211, 15], '1.')}
        with patch('exhad.secondary.decayer', decayer):
            for terminal in ('pythia', 'matched'):
                calls.clear()
                events = realize_rows({'xmldoc': 'xml'}, 4., rows, dict.fromkeys(rows, 2), 7, terminal)
                if terminal == 'pythia':
                    self.assertEqual(calls, [((), [pdgs] * 2, subseed(7, key)) for key, (pdgs, _) in rows.items()])
                    continue
                tau_seed = {key: subseed(subseed(7, key), 'spectator-tau') for key in rows}
                self.assertEqual(calls, [((), [[15]] * 2, tau_seed['2Pitau']), ((111, 310), [[411]] * 2, subseed(7, 'Dtau')),
                                         ((), [[15]] * 2, tau_seed['Dtau']), ((), [[211, 15]] * 2, subseed(7, 'Pitau'))])
                self.assertEqual([[int(p[5]) for p in event] for event in events['2Pitau']], [[211, 22, 22, 22]] * 2)
                self.assertEqual([[int(p[5]) for p in event] for event in events['Dtau']], [[22, 22]] * 2)



# Parton-pair thresholds restated for the independent reference (not imported from exhad).
REFERENCE_CUT = {1: .1396, 2: .1396, 3: .496, 4: 1.875, 5: 5.28}
V_A = 'E3*(mLLP-2*E3)'


def dalitz_integrals(expression, m, masses, pdgs, e1_edges, e3_edges, weight=None, nodes=8):
    """Integral of weight*max(|M|^2, 0) over each (E1, E3) cell of the physical region with the parton-pair cut.

    Independent of exhad sampling: its own sympy parse, Gauss-Legendre in E3 inside each E3 cell and in
    E1 between the analytic region limits at that E3 (boost of the (12) rest frame), clipped to the cell.
    """
    import sympy as sp
    e1s, e3s, ms, *particle_masses = sp.symbols('E1 E3 mLLP m1 m2 m3')
    f = sp.lambdify((ms, e1s, e3s, *particle_masses), sp.sympify(expression, locals=dict(E1=e1s, E3=e3s, mLLP=ms)), 'numpy')
    (m1, m2, m3), cut = masses, [abs(p) in REFERENCE_CUT for p in pdgs]
    top1 = top3 = math.inf
    if cut[0] and cut[1]:
        top3 = (m * m + m3 * m3 - (REFERENCE_CUT[abs(pdgs[0])] + REFERENCE_CUT[abs(pdgs[1])])**2) / (2 * m)
    elif cut[1] and cut[2]:
        top1 = (m * m + m1 * m1 - (REFERENCE_CUT[abs(pdgs[1])] + REFERENCE_CUT[abs(pdgs[2])])**2) / (2 * m)
    x, w = np.polynomial.legendre.leggauss(nodes)
    e1_edges, e3_edges = np.asarray(e1_edges, float), np.asarray(e3_edges, float)
    lo3, hi3 = e3_edges[:-1], np.minimum(e3_edges[1:], top3)
    half3 = np.clip(hi3 - lo3, 0., None)[:, None] / 2
    e3, w3 = (lo3[:, None] + half3 * (x + 1)).ravel(), (half3 * w).ravel()
    with np.errstate(divide='ignore', invalid='ignore'):
        s = m * m + m3 * m3 - 2 * m * e3
        open_, r = s > (m1 + m2)**2, np.sqrt(np.abs(s))
        star = (s + m1 * m1 - m2 * m2) / (2 * r)
        spread = np.sqrt(np.clip(e3 * e3 - m3 * m3, 0., None)) * np.sqrt(np.clip(star * star - m1 * m1, 0., None))
        lo1 = np.where(open_, ((m - e3) * star - spread) / r, 0.)
        hi1 = np.where(open_, np.minimum(((m - e3) * star + spread) / r, top1), 0.)
    a = np.maximum(lo1[None, :], e1_edges[:-1, None])
    half1 = np.clip(np.minimum(hi1[None, :], e1_edges[1:, None]) - a, 0., None)[..., None] / 2
    e1 = a[..., None] + half1 * (x + 1)
    g = np.maximum(np.broadcast_to(np.asarray(f(m, e1, e3[None, :, None], *masses), float), e1.shape), 0.)
    if weight is not None:
        g = g * weight(e1, e3[None, :, None])
    return ((g * half1 * w).sum(-1) * w3).reshape(len(e1_edges) - 1, len(e3_edges) - 1, nodes).sum(-1)


def chi2_p(observed, expected):
    """Pearson chi2 p-value against expected shape; cells expecting < 5 events are pooled."""
    observed = np.ravel(observed).astype(float)
    expected = np.ravel(expected) * observed.sum() / np.sum(expected)
    small = expected < 5
    if expected[small].sum() > 0:
        observed, expected = np.append(observed[~small], observed[small].sum()), np.append(expected[~small], expected[small].sum())
    elif observed[small].sum() > 0:
        return 0.
    else:
        observed, expected = observed[~small], expected[~small]
    from scipy import stats
    return stats.chisquare(observed, expected).pvalue


class LowMassRates(unittest.TestCase):
    """Exclusive table rows below the start masses: thresholds, gap rules and seams (no Pythia)."""

    def test_rates_close_below_start_and_raise_outside(self):
        from exhad import exclusive
        for model in ('dark-photon', 'alp-fermion', 'scalar-central', 'scalar-lower', 'scalar-upper', 'scalar-1809'):
            low, start = exclusive.table_start(model), exclusive.START[model]
            for mass in (low, *np.linspace(low, start, 23)[1:-1], np.nextafter(start, 0.)):
                p = exclusive.rates(model, float(mass))
                self.assertAlmostEqual(math.fsum(p), 1., places=14)
                self.assertTrue(p.min() >= 0 and not any(p[i] for i, row in enumerate(exclusive.rows(model)) if row.partonic))
                self.assertFalse(any(v > 0 and row.threshold >= mass for v, row in zip(p, exclusive.rows(model))
                                     if not row.partonic), (model, mass))
                labels, _, q = boson_rates(model, float(mass), 'central')
                self.assertAlmostEqual(q[0], math.fsum(v for v, row in zip(p, exclusive.rows(model)) if row.has_hadrons), 14)
            for mass in (low * (1 - 1e-6), start):
                with self.assertRaises(ValueError):
                    exclusive.rates(model, mass)
        with self.assertRaisesRegex(ValueError, '0.003 <= mass'):
            boson_rates('dark-photon', .0029, 'central')
        scalar = {row.label: v for row, v in zip(exclusive.rows('scalar-central'), exclusive.rates('scalar-central', .993))}
        self.assertEqual((scalar['KLKL'], scalar['KSKS']), (0., 0.))  # nodes below 2 m(K0) are zeroed by the threshold rule

    def test_gap_rules_join_the_matched_side(self):
        from exhad import exclusive
        from exhad.models import model_info
        np.testing.assert_array_equal(exclusive.rates('dark-photon', 1.6995), exclusive.rates('dark-photon', 1.699))
        rows = exclusive.rows('alp-fermion')
        held, bridged = exclusive.rates('alp-fermion', 1.91), exclusive.rates('alp-fermion', 1.9105)
        hadronic = [row.has_hadrons for row in rows]
        self.assertAlmostEqual(math.fsum(held[hadronic]), math.fsum(bridged[hadronic]), places=14)
        self.assertNotEqual(held[[row.label for row in rows].index('ppbar')], bridged[[row.label for row in rows].index('ppbar')])
        for model in ('scalar-central', 'scalar-1809'):  # the two-meson and nucleon rows reach the outer owner at 2 GeV
            owners = model_info(model)['owner_probabilities']
            at_owner = {label: float(np.interp(2., owners['masses'], v)) for label, v in owners['probabilities'].items()}
            below = {row.label: v for row, v in zip(exclusive.rows(model), exclusive.rates(model, 2. - 1e-12))}
            for label in ('PipPim', '2Pi0', '2Kch', 'KLKL', 'KSKS', 'ppbar', 'nnbar'):
                self.assertAlmostEqual(below[label], at_owner[label], places=9)
            self.assertAlmostEqual(below['2Pip2Pim'] + below['PipPim2Pi0'], at_owner['Jets-GG'] + at_owner['Jets-ss'], places=9)

    def test_baryon_width_joins_and_kappa_rows(self):
        from exhad.b_l import low_mass, rates
        m_j = low_mass.join_mass()
        self.assertAlmostEqual(m_j, low_mass.matching_mass(), places=9)
        for mass in (m_j, 2.):
            self.assertAlmostEqual(low_mass.hadronic_width(np.nextafter(mass, 0.)) / rates.total_width(mass), 1., places=6)
        for mass in np.arange(1.737, 2., .0137):
            self.assertGreaterEqual(rates.total_width(mass) - math.fsum(low_mass.row_widths(mass)), 0.)
        from exhad.models import model_info
        info = model_info('b-l')
        self.assertEqual((info['support_gev'], info['exclusive_gev']), ([2.0, 5.0], [0.001, 2.0]))

    def test_three_pion_rows_share_the_bose_symmetric_expression(self):
        from exhad import exclusive
        from exhad.core.kinematics import matrix_element
        from exhad.core.pdg import m0
        tables = [{row.label: row for row in exclusive.rows(model)}['Pip_Pim_Pi0'] for model in ('dark-photon', 'b-l')]
        self.assertEqual(tables[0].expression, tables[1].expression)
        f = matrix_element(tables[0].expression, tuple(m0(p) for p in tables[0].pdgs))
        for mass in (.45, .78, 1.02, 1.5):  # near-symmetric Dalitz points, inside the physical region
            e1, e3 = mass / 3 + .005, mass / 3
            e2 = mass - e1 - e3
            self.assertGreater(f(mass, e1, e3), 0.)
            # pi+ <-> pi- exchange, up to the roundoff of the compiled expression (about 4e-12 relative here)
            self.assertAlmostEqual(f(mass, e2, e3) / f(mass, e1, e3), 1., places=10)

    def test_hnl_two_lepton_rows_label_the_mixing_neutrino(self):
        for alpha, label, pdgs in ((0, 'emuv', [11, 14, -13]), (1, 'emuv', [11, -12, -13]), (2, 'mutauv', [13, -14, -15])):
            labels, ids, _, p = hnl_rates(3., np.eye(3)[alpha])
            self.assertIn(pdgs, [i for l, i, v in zip(labels, ids, p) if l == label and v > 0])


class ThreeBody(unittest.TestCase):
    """Three-body primaries follow phase space times max(|M|^2, 0) at every batch size."""

    @staticmethod
    def row(label, alpha):
        return next(r[5 + alpha] for r in hnl_tables()[0] if r[0] == label)

    def assert_matches_quadrature(self, events, expression, m, pdgs, bins=40):
        """E1 and E3 marginal chi2 and mean pulls against the quadrature; returns (<E1>, <E3>) references."""
        from exhad.full_decays import QUARK_MASSES
        masses = [QUARK_MASSES.get(abs(p), m0(p)) for p in pdgs]
        e1, e3 = events[:, 0, 3], events[:, 2, 3]
        hi1, hi3 = (m * m + masses[0]**2 - (masses[1] + masses[2])**2) / (2 * m), (m * m + masses[2]**2 - (masses[0] + masses[1])**2) / (2 * m)
        fine1, fine3 = np.linspace(masses[0], hi1, 16 * bins + 1), np.linspace(masses[2], hi3, 16 * bins + 1)
        norm = dalitz_integrals(expression, m, masses, pdgs, fine1[[0, -1]], fine3)
        for sample, edges, cells in ((e3, fine3, norm.reshape(bins, 16).sum(1)),
                                     (e1, fine1, dalitz_integrals(expression, m, masses, pdgs, fine1[::16], fine3).sum(1))):
            self.assertGreaterEqual(chi2_p(np.histogram(sample, edges[::16])[0], cells), 1e-3)
        means = [dalitz_integrals(expression, m, masses, pdgs, fine1[[0, -1]], fine3, weight=lambda a, b, k=k: (a, b)[k]).sum() / norm.sum()
                 for k in (0, 1)]
        for sample, mean in zip((e1, e3), means):
            self.assertLess(abs(sample.mean() - mean) / (sample.std() / math.sqrt(len(sample))), 4.5)
        return means

    def test_v_a_three_neutrino_law(self):
        from scipy import stats
        for m, seed in ((.1, 9000001), (1., 9000002), (5., 9000003)):
            a = primary(m, [12, 14, -14], 60000, seed, V_A).reshape(-1, 3, 8)
            n, x1, x3 = len(a), 2 * a[:, 0, 3] / m, 2 * a[:, 2, 3] / m
            np.testing.assert_allclose(a[:, :, :4].sum(axis=1), np.tile([0., 0., 0., m], (n, 1)), atol=1e-9 * m)
            self.assertGreaterEqual(stats.kstest(x3, stats.beta(3, 2).cdf).pvalue, 1e-3)
            self.assertGreaterEqual(stats.kstest(x1, lambda y: np.clip(2 * y**3 - y**4, 0, 1)).pvalue, 1e-3)
            self.assertLess(abs(x3.mean() - .6), 4 * .2 / math.sqrt(n))
            self.assertLess(abs(x3.var() - .04), 4 * math.sqrt((3.771e-3 - .04**2) / n))
            self.assertLess(abs(x1.mean() - .7), 4 * math.sqrt(.0433 / n))
            edges = np.linspace(0., m / 2, 21)
            cells = dalitz_integrals(V_A, m, (0., 0., 0.), (12, 14, -14), edges, np.linspace(0., m / 2, 321)).reshape(20, 20, 16).sum(-1)
            self.assertGreaterEqual(chi2_p(np.histogram2d(a[:, 0, 3], a[:, 2, 3], [edges, edges])[0], cells), 1e-3)
            cos = a[:, 0, 2] / np.linalg.norm(a[:, 0, :3], axis=1)
            self.assertLess(abs(cos.mean()), 4 * math.sqrt(1 / (3 * n)))
            self.assertLess(abs((cos**2).mean() - 1 / 3), 4 * math.sqrt(4 / (45 * n)))

    def test_hnl_rows_match_independent_quadrature(self):
        cases = (('2ev', 0, [11, 12, -11], 2.), ('2muv', 1, [13, 14, -13], .5), ('emuv', 0, [11, 14, -13], .12),
                 ('2tauv', 2, [15, 16, -15], 4.), ('Jets-ude', 0, [2, -1, 11], 2.), ('Jets-ude', 0, [2, -1, 11], 5.),
                 ('Jets-udtau', 2, [2, -1, 15], 3.), ('Jets-cde', 0, [4, -1, 11], 3.5), ('Jets-ccv', 0, [4, -4, 12], 6.),
                 ('2Pie', 0, [211, 111, 11], 1.845), ('2Piv', 0, [211, -211, 12], 1.375), ('2Pitau', 2, [211, 111, 15], 3.61))
        published = {('Jets-cde', 3.5): (.9763, .4714), ('2Pie', 1.845): (.5963, .8102), ('2muv', .5): (.7421, .6978)}
        for i, (label, alpha, pdgs, m) in enumerate(cases):
            with self.subTest(row=label, mass=m):
                expression = self.row(label, alpha)
                events = primary(m, pdgs, 40000, 9000100 + i, expression).reshape(-1, 3, 8)
                means = self.assert_matches_quadrature(events, expression, m, pdgs)
                if (label, m) in published:  # reference means from an independent quadrature script
                    np.testing.assert_allclose(2 * np.array(means) / m, published[label, m], atol=1e-4)
                if abs(pdgs[0]) in REFERENCE_CUT and abs(pdgs[1]) in REFERENCE_CUT:
                    pair = events[:, 0, :4] + events[:, 1, :4]
                    w = np.sqrt(pair[:, 3]**2 - (pair[:, :3]**2).sum(1))
                    self.assertGreaterEqual(w.min(), REFERENCE_CUT[abs(pdgs[0])] + REFERENCE_CUT[abs(pdgs[1])] - 1e-9)

    def test_batch_size_invariance(self):
        from scipy import stats
        for m, pdgs, expression in ((1., [12, 14, -14], V_A), (2., [2, -1, 11], self.row('Jets-ude', 0))):
            arms = []
            for count in (1, 64, 512):
                events = np.concatenate([primary(m, pdgs, count, 9000200 + count * 10000 + i, expression)
                                         for i in range(8192 // count)]).reshape(-1, 3, 8)
                self.assert_matches_quadrature(events, expression, m, pdgs, bins=20)
                arms.append(events[:, 2, 3])
            for i, j in ((0, 1), (0, 2), (1, 2)):
                self.assertGreaterEqual(stats.ks_2samp(arms[i], arms[j]).pvalue, 1e-3)
            one = primary(m, pdgs, 50000, 9000299, expression).reshape(-1, 3, 8)
            self.assertEqual(len({(e1, e3) for e1, e3 in one[:, [0, 2], 3]}), 50000)  # no resampled duplicates

    def test_matrix_element_policy(self):
        from exhad.core import kinematics
        for expression in ('-1.', '-' + V_A, '0.'):
            with self.assertRaisesRegex(ValueError, 'not positive anywhere'):
                primary(1., [12, 14, -14], 4, 1, expression)
        clipped = primary(1., [12, 14, -14], 40000, 9000301, V_A + '-1e-9*mLLP**3').reshape(-1, 3, 8)
        self.assertLess(abs(2 * clipped[:, 2, 3].mean() - .6), 4 * .2 / math.sqrt(40000))
        flat = primary(2., [13, 2, -3], 40000, 9000302, '1.').reshape(-1, 3, 8)
        self.assert_matches_quadrature(flat, '1', 2., [13, 2, -3])
        with self.assertRaisesRegex(ValueError, 'closed'):  # c s-bar pair needs 2.371 GeV; a closed channel fails at once
            primary(2.3, [11, 4, -3], 4, 1, '1.')
        function, bound, acceptance = kinematics.envelope(V_A, 1., (12, 14, -14), (0., 0., 0.))
        with patch.object(kinematics, 'envelope', return_value=(function, .5 * bound, acceptance)), \
                self.assertRaisesRegex(RuntimeError, 'envelope'):
            primary(1., [12, 14, -14], 1000, 1, V_A)
        with patch.object(kinematics, 'envelope', return_value=(function, 1e6 * bound, 1.)), \
                self.assertRaisesRegex(RuntimeError, 'proposal cap'):
            primary(1., [12, 14, -14], 1000, 1, V_A)

    def test_negative_matrix_elements_sample_the_positive_part(self):
        """|M|^2 negative over much of the region is clipped, not rejected (negative everywhere raises above)."""
        half = primary(1., [12, 14, -14], 40000, 9000303, 'E1-0.3*mLLP').reshape(-1, 3, 8)  # negative below E1 = 0.3
        self.assertGreater(half[:, 0, 3].min(), .3)
        self.assert_matches_quadrature(half, 'E1-0.3*mLLP', 1., [12, 14, -14])

    def test_leptonic_rows_use_the_generated_masses(self):
        """|M|^2 of the leptonic rows vanishes on the generated boundary and is positive inside, even just above threshold.

        Each row is tested at two masses just above its generated threshold, the second within 0.13 MeV of it.
        """
        from exhad.core import kinematics
        for i, (label, alpha, pdgs, m) in enumerate((('emuv', 0, [11, 14, -13], .1068777), ('emuv', 0, [11, 14, -13], .1063),
                                                     ('mutauv', 2, [13, 16, -15], 1.8825341), ('mutauv', 2, [13, 16, -15], 1.88249),
                                                     ('etauv', 2, [11, 16, -15], 1.7773898), ('etauv', 2, [11, 16, -15], 1.77734))):
            with self.subTest(row=label, mass=m):
                expression, masses = self.row(label, alpha), tuple(m0(p) for p in pdgs)
                function = kinematics.matrix_element(expression, masses)
                kinematics.seed(9000320 + i)
                e1, e3 = kinematics.dalitz(m, *masses, 100000, pdgs)
                values = np.asarray(function(m, e1, e3), float)
                self.assertGreater(values.min(), 0.)
                hi1, hi3 = ((m * m + masses[k]**2 - (masses[1] + masses[2 - k])**2) / (2 * m) for k in (0, 2))
                edge = min(abs(function(m, hi1, (masses[2] + hi3) / 2)), abs(function(m, (masses[0] + hi1) / 2, hi3)))
                self.assertLessEqual(edge, 1e-9 * values.max())
                events = primary(m, pdgs, 40000, 9000310 + i, expression).reshape(-1, 3, 8)
                self.assert_matches_quadrature(events, expression, m, pdgs)

    def test_envelope_bounds_shipped_rows(self):
        from exhad.full_decays import QUARK_MASSES
        from exhad.core import kinematics
        kinematics.seed(9000400)
        for row in hnl_tables()[0]:
            pdgs = tuple(int(p) for p in row[1] if int(p) != -999)
            if len(pdgs) != 3:
                continue
            masses = tuple(QUARK_MASSES.get(abs(p), m0(p)) for p in pdgs)
            cut = [REFERENCE_CUT.get(abs(p)) for p in pdgs]
            threshold = (cut[0] + cut[1] + masses[2] if cut[0] and cut[1] else
                         cut[1] + cut[2] + masses[0] if cut[1] and cut[2] else sum(masses))
            for alpha in range(3):
                positive = [m for m, br in row[2 + alpha] if br > 0]  # from the first tabulated rate up
                for m in positive[::max(1, len(positive) // 4)]:
                    with self.subTest(row=row[0], alpha=alpha, mass=m):
                        if m <= threshold:  # Jets-cbe at 7.15 GeV: BR 1e-16 but no phase space; reported as closed
                            self.assertRaisesRegex(ValueError, 'closed', kinematics.envelope, row[5 + alpha], m, pdgs, masses)
                            continue
                        function, bound, _ = kinematics.envelope(row[5 + alpha], m, pdgs, masses)
                        e1, e3 = kinematics.dalitz(m, *masses, 20000, pdgs)
                        values = np.broadcast_to(np.asarray(function(m, e1, e3), float), e1.shape)
                        self.assertLessEqual(values.max(), bound / 1.03)


MPI, MPI0, MK = .13957, .13498, .49368


def lips_reference(m, masses, size, rng):
    """Brute-force flat-LIPS points independent of the Raubold-Lynch chain: (momenta [size, n, 4], weights).

    Four bodies: particles 0 and 1 uniform in momentum balls, the 2-3 pair isotropic in its rest frame, weight
    p*/m_23 / (E_0 E_1). Six bodies: two clusters 012 and 345 with uniform masses, each a uniform Dalitz point in
    its own rest frame with a Haar orientation, weight p*(m; m_A, m_B) m_A m_B area_A area_B.
    """
    from scipy.spatial.transform import Rotation
    mu, n = np.asarray(masses), len(masses)

    def pstar(a, b, c):
        return np.sqrt(np.maximum((a * a - (b + c)**2) * (a * a - (b - c)**2), 0.)) / (2 * np.maximum(a, 1e-300))

    def boost(p, frame):  # p given in the rest frame of the lab four-vector frame
        mass = np.sqrt(np.maximum(frame[:, 3]**2 - (frame[:, :3]**2).sum(1), 0.))[:, None]
        dot = (frame[:, None, :3] * p[:, :, :3]).sum(2)
        out = p.copy()
        out[:, :, :3] += frame[:, None, :3] * ((dot / (frame[:, None, 3] + mass) + p[:, :, 3]) / mass)[:, :, None]
        out[:, :, 3] = (frame[:, None, 3] * p[:, :, 3] + dot) / mass
        return out

    def isotropic_pair(mass, m1, m2):
        c, phi = rng.uniform(-1, 1, len(mass)), rng.uniform(0, 2 * np.pi, len(mass))
        q = pstar(mass, m1, m2)[:, None] * np.column_stack([np.sqrt(1 - c * c) * np.cos(phi), np.sqrt(1 - c * c) * np.sin(phi), c])
        e1 = (mass**2 + m1**2 - m2**2) / (2 * np.maximum(mass, 1e-300))
        return np.stack([np.column_stack([q, e1]), np.column_stack([-q, mass - e1])], axis=1)

    def dalitz(mass, m1, m2, m3):
        top1, top3 = (mass**2 + m1**2 - (m2 + m3)**2) / (2 * mass), (mass**2 + m3**2 - (m1 + m2)**2) / (2 * mass)
        e1, e3 = m1 + (top1 - m1) * rng.random(len(mass)), m3 + (top3 - m3) * rng.random(len(mass))
        e2, a1, a3 = mass - e1 - e3, e1**2 - m1**2, e3**2 - m3**2
        ok = (e2 > m2) & ((e2**2 - m2**2 - a1 - a3)**2 < 4 * a1 * a3) & (top1 > m1) & (top3 > m3)
        a1, a3 = np.where(ok, a1, 1.), np.where(ok, a3, 1.)
        cos13 = np.where(ok, (e2**2 - m2**2 - a1 - a3) / (2 * np.sqrt(a1 * a3)), 0.)
        p1 = np.column_stack([np.zeros_like(e1), np.zeros_like(e1), np.sqrt(a1)])
        p3 = np.sqrt(a3)[:, None] * np.column_stack([np.sqrt(1 - cos13**2), np.zeros_like(e1), cos13])
        spin = Rotation.random(len(mass), random_state=int(rng.integers(2**31)))
        p1, p3 = spin.apply(p1), spin.apply(p3)
        rows = np.stack([np.column_stack([p1, e1]), np.column_stack([-p1 - p3, e2]), np.column_stack([p3, e3])], axis=1)
        return rows, ok * np.maximum(top1 - m1, 0) * np.maximum(top3 - m3, 0)

    if n == 4:
        radius = np.array([pstar(np.array([m]), mu[i], mu.sum() - mu[i])[0] for i in range(2)])
        d = rng.normal(size=(size, 2, 3))
        p = d / np.linalg.norm(d, axis=2)[:, :, None] * (radius[None, :, None] * rng.random((size, 2, 1))**(1 / 3))
        e = np.sqrt((p * p).sum(2) + mu[:2]**2)
        pair = np.column_stack([-p.sum(1), m - e.sum(1)])
        s = pair[:, 3]**2 - (pair[:, :3]**2).sum(1)
        ok = (pair[:, 3] > 0) & (s > (mu[2] + mu[3])**2)
        pair[~ok] = [0., 0., 0., 1.]
        root = np.sqrt(pair[:, 3]**2 - (pair[:, :3]**2).sum(1))
        weights = ok * pstar(root, mu[2], mu[3]) / root / e.prod(1)
        return np.concatenate([np.concatenate([p, e[:, :, None]], 2), boost(isotropic_pair(root, mu[2], mu[3]), pair)], 1), weights
    low_a, low_b = mu[:3].sum(), mu[3:].sum()
    mass_a, mass_b = low_a + (m - low_b - low_a) * rng.random(size), low_b + (m - low_a - low_b) * rng.random(size)
    ok = mass_a + mass_b < m
    mass_a, mass_b = np.where(ok, mass_a, low_a + 1e-3), np.where(ok, mass_b, low_b + 1e-3)
    frames = isotropic_pair(np.full(size, float(m)), mass_a, mass_b)
    a, area_a = dalitz(mass_a, *mu[:3])
    b, area_b = dalitz(mass_b, *mu[3:])
    weights = ok * pstar(np.full(size, float(m)), mass_a, mass_b) * mass_a * mass_b * area_a * area_b
    return np.concatenate([boost(a, frames[:, 0]), boost(b, frames[:, 1])], 1), weights


def weighted_chi2_p(sample, reference, weights, bins=30):
    """chi2 p-value of an unweighted sample against a weighted reference with its own statistical error."""
    from scipy import stats
    edges = np.quantile(sample, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = min(sample.min(), reference.min()) - 1e-12, max(sample.max(), reference.max()) + 1e-12
    n = np.histogram(sample, edges)[0]
    w, w2 = np.histogram(reference, edges, weights=weights)[0], np.histogram(reference, edges, weights=weights**2)[0]
    x = (n / n.sum() - w / w.sum())**2 / (n / n.sum()**2 + w2 / w.sum()**2)
    return stats.chi2.sf(x.sum(), bins - 1)


class NBody(unittest.TestCase):
    """Flat n-body primaries: exact closure, a certified weight bound, and flat-LIPS projections."""
    CASES = (('4pi', [211, -211, 111, 111], [MPI, MPI, MPI0, MPI0], (.60, 1.69)),
             ('KKpipi', [321, -321, 211, -211], [MK, MK, MPI, MPI], (1.32, 1.69)),
             ('6pi', [211, 211, -211, -211, 111, 111], [MPI] * 4 + [MPI0] * 2, (.90, 1.69)))

    def test_projections_match_brute_force_phase_space(self):
        rng = np.random.default_rng(20260915)
        for label, pdgs, masses, points in self.CASES:
            for m in points:
                events = primary(m, pdgs, 200000, 9000401).reshape(-1, len(pdgs), 8)
                parts = [lips_reference(m, masses, 500000, rng) for _ in range(3)]
                reference, weights = np.concatenate([p for p, _ in parts]), np.concatenate([w for _, w in parts])
                self.assertGreaterEqual(np.count_nonzero(weights), 1e5)
                for name, key in (('m234', lambda x: x[:, 1:4]), ('m34', lambda x: x[:, 2:4]),
                                  ('E1', lambda x: x[:, :1]), ('En', lambda x: x[:, -1:])):
                    def project(x, key=key, name=name):
                        s = key(x)[:, :, :4].sum(1)
                        return s[:, 3] if name[0] == 'E' else np.sqrt(np.maximum(s[:, 3]**2 - (s[:, :3]**2).sum(1), 0))
                    with self.subTest(row=label, mass=m, projection=name):
                        self.assertGreaterEqual(weighted_chi2_p(project(events), project(reference), weights), 1e-4)

    def test_closure_bound_prefix_and_three_body_law(self):
        from scipy import stats
        from exhad.core import kinematics
        for _, pdgs, masses, points in self.CASES:
            for m in (sum(masses) + 1e-3, *points):
                events = primary(m, pdgs, 20000, 9000402).reshape(-1, len(pdgs), 8)
                self.assertLess(np.abs(events[:, :, :4].sum(1) - [0, 0, 0, m]).max(), 1e-12)
                self.assertLess(np.abs(events[:, :, 3]**2 - (events[:, :, :3]**2).sum(2) - events[:, :, 4]**2).max(), 1e-12)
                self.assertTrue(np.all(events[:, :, 5] == pdgs) and len(np.unique(events[:, 0, 3])) == 20000)
                # a count-independent stream: every batch is the prefix of a larger one
                self.assertTrue(np.array_equal(primary(m, pdgs, 777, 9000402).reshape(-1, len(pdgs), 8), events[:777]))
                cum, T = np.cumsum(masses), m - sum(masses)
                split = np.sort(np.random.default_rng(1).random((200000, len(masses) - 2)), axis=1) * T
                M = np.column_stack([np.full(len(split), masses[0]), cum[1:-1] + split, np.full(len(split), m)])
                t = np.diff(np.column_stack([split, np.full(len(split), T)]), axis=1, prepend=0.)
                largest = kinematics._breakup(M[:, 1:], M[:, :-1], np.array(masses[1:]), t).prod(1).max()
                bound = kinematics.phase_space_bound(float(m), tuple(masses))
                self.assertTrue(largest <= bound <= largest * (1.02 if len(masses) == 4 else 1.25))
        with self.assertRaisesRegex(ValueError, 'closed'):
            primary(4 * MPI, [211, -211, 211, -211], 1, 1)
        with self.assertRaisesRegex(ValueError, 'flat phase space'):
            primary(1., [211, -211, 211, -211], 1, 1, V_A)
        with patch.object(kinematics, 'phase_space_bound', return_value=1e-9), self.assertRaisesRegex(RuntimeError, 'exceeds its bound'):
            primary(1., [211, -211, 211, -211], 10, 1)
        pdgs, m = [211, -211, 111], 1.1
        kinematics.seed(9000403)
        flat = kinematics.n_body(m, 100000, pdgs, [MPI, MPI, MPI0], [1, -1, 0], [1, 1, 1]).reshape(-1, 3, 8)
        dalitz = primary(m, pdgs, 100000, 9000404, '1.').reshape(-1, 3, 8)
        for k in (0, 2):
            self.assertGreaterEqual(stats.ks_2samp(flat[:, k, 3], dalitz[:, k, 3]).pvalue, 1e-3)

    def test_matrix_element_real_part(self):
        from exhad.core.kinematics import matrix_element
        tiny = matrix_element('E1*(1 + 1e-15*I)', (0., 0., 0.))
        self.assertEqual(tiny(1., np.array([.2, .3]), np.array([.1, .1])).tolist(), [.2, .3])
        with self.assertRaisesRegex(ValueError, 'imaginary'):
            matrix_element('E1*(1 + 1e-9*I)', (0., 0., 0.))(1., np.array([.2]), np.array([.1]))


class Transition(unittest.TestCase):
    def test_quintic_mixture(self):
        self.assertEqual([pythia_fraction(m) for m in (3., 4., 4.5, 5., 20.)], [0., 0., .5, 1., 1.])
        self.assertTrue(np.all(np.diff(pythia_fraction(np.linspace(4., 5., 1001))) >= 0))
        for epsilon in (.001, .0001):  # value, slope and curvature join at both ends
            self.assertLess(pythia_fraction(4. + epsilon), 11 * epsilon**3)
            self.assertLess(1 - pythia_fraction(5. - epsilon), 11 * epsilon**3)

    def test_boost_round_trip(self):
        rows = np.array([[.7, .2, .5, 1.2, math.sqrt(1.2**2 - .7**2 - .2**2 - .5**2), 421.]])
        shifted = np.array(boost(rows, [3., -2., 4., 10.]))
        np.testing.assert_allclose(boost(shifted, [3., -2., 4., 10.], inverse=True), rows, atol=1e-14)
        np.testing.assert_allclose(shifted[:, 3]**2 - (shifted[:, :3]**2).sum(axis=1), rows[:, 4]**2)
        np.testing.assert_array_equal(shifted[:, 4:], rows[:, 4:])


class AlpFermionCharm(unittest.TestCase):
    def test_open_charm_fraction(self):
        fraction = lambda m: charm.open_charm_fraction(m, 1.86486, 2.00698)
        self.assertEqual([fraction(3.8717), fraction(1.86486 + 2.00698), fraction(4.), fraction(5.)], [0., 0., 1., 1.])
        self.assertTrue(0. < fraction(3.872) < fraction(3.99) < 1.)
        self.assertAlmostEqual(fraction(3.95), ((3.95**2 - 3.87184**2) * (3.95**2 - .14212**2) / (4 * 3.95**2)
                               / ((16 - 3.87184**2) * (16 - .14212**2) / 64))**1.5, places=12)

    def runtime(self, events):
        runtime = object.__new__(charm.AlpFermionCharmRuntime)
        runtime.component_for = lambda mass, seed: 'glue'
        runtime.generate_component = lambda *args: next(events)
        return runtime

    def test_eta_dipion_isospin_projection(self):
        for eta in (221, 331):
            charged, neutral, counts = (eta, 211, -211), (eta, 111, 111), Counter()
            for seed in range(6000):
                runtime = self.runtime(iter((charged, (321, -321, 111), neutral)))
                with patch.object(charm, 'derive_ownership_cut', lambda e: SimpleNamespace(pdg_ids=e)):
                    output_event = runtime.generate(3.75, seed)
                expected = (charm.splitmix64(seed ^ charm._ISOSPIN_TAG) >> 11) / float(1 << 53) < 2 / 3
                self.assertEqual(output_event, charged if expected else neutral)
                counts[expected] += 1
            self.assertLess(abs(counts[True] - 4000), 250)
        runtime = self.runtime(iter(((321, -321, 111),)))
        with patch.object(charm, 'derive_ownership_cut', lambda e: SimpleNamespace(pdg_ids=e)):
            self.assertEqual(runtime.generate(3.75, 91), (321, -321, 111))

    def test_numerical_retry_keeps_component(self):
        runtime = object.__new__(charm.AlpFermionCharmRuntime)
        event, worker_mock = SimpleNamespace(primary_pdgs=(221, 211, -211)), Mock()
        worker_mock.request.return_value = event
        runtime.pool, runtime.catalog = charm.WorkerPool(8), None
        runtime.pool.items['glue'] = worker_mock
        with patch.object(charm, 'validate_complete_event',
                          side_effect=[charm.EventModelError('event does not conserve px'), None]):
            self.assertIs(runtime.generate_component(3.75, 85336, 'glue'), event)
        self.assertEqual(worker_mock.request.call_count, 2)


class BLOpenCharm(unittest.TestCase):
    def generator(self):
        g = bl.BaryonOpenCharmGenerator.__new__(bl.BaryonOpenCharmGenerator)
        g.light = Mock()
        g.light.rate_point.return_value = SimpleNamespace(inclusive_width_per_g_b_squared_gev=.1)
        g.light.sample.side_effect = lambda mass, count, **kw: SimpleNamespace(events=(('light',),) * count)
        g.charm, g._charm_event = build_open_charm_owner(), Mock(return_value=('charm',))
        return g

    def test_charge_factor_quarter(self):
        self.assertEqual(bl.CHARM_CURRENT_WIDTH_SCALE, .25)
        point = bl.BaryonOpenCharmRatePoint(SimpleNamespace(inclusive_width_per_g_b_squared_gev=.1),
                                            SimpleNamespace(inclusive_charm_width=.04))
        self.assertAlmostEqual(point.charm_width_per_g_b_squared_gev, .01)
        self.assertAlmostEqual(point.charm_probability, 1 / 11)

    def test_light_sample_unchanged_below_threshold(self):
        g = self.generator()
        with patch.object(g.charm, 'closure_at', side_effect=AssertionError('charm lookup below threshold')):
            for mass in (2., 3., 3.729, g.charm.open_charm_threshold_gev):
                batch = g.sample(mass, 10, seed=732, variation='p-high')
                self.assertEqual((batch.events, batch.rate_point.charm_probability), ((('light',),) * 10, 0.))
                g.light.sample.assert_called_with(mass, 10, seed=732, variation='p-high', active_pool=None)
        g._charm_event.assert_not_called()

    def test_outer_draw_is_binomial_and_repeatable(self):
        g = self.generator()
        batch = g.sample(5., 100000, seed=42)
        self.assertEqual(batch.events, g.sample(5., 100000, seed=42).events)
        n, p = len(batch.events), batch.rate_point.charm_probability
        self.assertLess(abs(batch.origins.count('open-charm') - n * p), 6 * math.sqrt(n * p * (1 - p)))
        self.assertEqual(batch.origins.count('open-charm'), batch.events.count(('charm',)))
        for mass, count, seed in ((1.9, 1, 0), (5.01, 1, 0), (float('nan'), 1, 0), (4., -1, 0), (4., 1, -1), (4., True, 0)):
            with self.assertRaises((ValueError, TypeError)):
                g.sample(mass, count, seed=seed)


class DarkPhotonCharmFamilies(unittest.TestCase):
    def test_family_draw_independent_of_light_charm_draw(self):
        # The family uniform is independent of the owner uniform, which charm events only see above p_light;
        # sharing it would put every 3.80-GeV charm event in the residual instead of ddbar 0.69 / residual 0.31.
        from exhad.core.seeds import splitmix64
        from exhad.dark_photon import open_charm as dark_photon
        from exhad.dark_photon.closure import load_inputs
        tags = [value for name, value in vars(dark_photon).items() if name.endswith('_TAG')]
        self.assertEqual(len(tags), len(set(tags)))
        mass, count, seed = 3.8, 40000, 7
        inputs, owner = load_inputs(), build_open_charm_owner()
        light = SimpleNamespace(inputs=inputs, post_edge_forbidden_keys=frozenset(), generate=lambda *args: None)
        backend = SimpleNamespace(resolved_event=lambda m, pdgs, s: SimpleNamespace(primary_pdgs=tuple(pdgs), ddbar=True),
                                  unresolved_event=lambda m, s: SimpleNamespace(primary_pdgs=(), ddbar=False))
        with patch.object(dark_photon, 'validate_complete_event'), patch.object(dark_photon, 'ownership_key'):
            events = dark_photon.generate_total_em_dark_photon_events(
                light, SimpleNamespace(owner=owner, backend=backend, catalog=None), mass, count, seed)
        point = owner.closure_at(mass)
        (ddbar,) = [item for item in point.resolved if item.sigma_nb > 0.]
        p = ddbar.width / point.inclusive_charm_width
        p_light = inputs.post_edge.closure_at(mass).inclusive_width_gev / (
            inputs.post_edge.closure_at(mass).inclusive_width_gev + point.inclusive_charm_width)
        halves = ([], [])
        for index, event in enumerate(events):
            u = dark_photon._unit_interval(splitmix64(seed ^ splitmix64(index)), dark_photon._OWNER_DRAW_TAG)
            self.assertEqual(event is not None, u >= p_light)  # the owner uniform alone decides light versus charm
            if event is not None:
                halves[u >= (1. + p_light) / 2].append(event.ddbar)
        for half in (halves[0], halves[1], halves[0] + halves[1]):
            self.assertLess(abs(sum(half) - len(half) * p), 4.5 * math.sqrt(len(half) * p * (1 - p)))


class AlpFermionChargeMatching(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matching = ConditionalChannelMatching(json.loads((DATA / 'alp-fermion/model1.json').read_text())['conditional_matching'])

    def test_eta_partitions_are_two_to_one_and_keep_group_totals(self):
        entry = self.matching.families['eta-pipi']
        for mass in (3., 3.001, 3.2, 3.5, 3.75, 4., 4.5, 5.):
            proposal, _ = self.matching._proposal(entry, mass)
            weights, _ = self.matching.at(mass)
            target = {key: value * weights['eta-pipi'][key] for key, value in proposal.items()}
            self.assertAlmostEqual(math.fsum(target.values()), 1., places=12)
            for eta in (221, 331):
                charged, neutral = f'-211:1 211:1 {eta}:1', f'111:2 {eta}:1'
                self.assertAlmostEqual(target[charged] / target[neutral], 2., places=12)
                self.assertAlmostEqual(target[charged] + target[neutral], proposal[charged] + proposal[neutral], places=12)

    def test_exclusive_boundary_is_retained(self):
        self.assertEqual(self.matching.start, 1.911)
        weights, _ = self.matching.at(self.matching.start)
        for family, entry in self.matching.families.items():
            proposal, _ = self.matching._proposal(entry, self.matching.start)
            for key, expected in entry['left'].items():
                self.assertAlmostEqual(proposal[key] * weights[family][key], expected, places=12)


class PublicInterface(unittest.TestCase):
    def test_model_names(self):
        for constructor in (validate_model, Generator):
            with self.assertRaisesRegex(ValueError, 'choose from: dark-photon, alp-fermion, scalar-central, '
                                        'scalar-lower, scalar-upper, scalar-1809, b-l, hnl$'):
                constructor('unknown')

    def test_accelerator_scope(self):
        config = {'conditional_acceleration': 'unused'}
        base = dict(model='alp-fermion', mass=3., events=1, seed=1, variation='central')
        with patch.object(worker, '_sample_reference') as reference, \
                patch.object(worker, '_conditional_pool', return_value='fast'):
            def pool(config, request):
                worker.sample(config, request)
                return reference.call_args.kwargs.get('active_pool')
            self.assertEqual(pool(config, base), 'fast')
            self.assertEqual(pool(config, dict(base, mass=4.)), 'fast')
            for fields in (dict(execution='reference'), dict(weighted=True), dict(mass=2.4),
                           dict(model='dark-photon'), dict(model='scalar-central', mass=2.99), dict(model='hnl'),
                           dict(model='scalar-1809', execution='reference')):
                self.assertIsNone(pool(config, dict(base, **fields)))
            for model in ('scalar-central', 'scalar-lower', 'scalar-upper', 'scalar-1809'):  # rate-first four-pion
                self.assertEqual(pool(config, dict(base, model=model)), 'fast')
                self.assertEqual(worker._conditional_pool.call_args.args[1:], ('scalar',))
            for mass in (2., 3.5, 5.):  # the vector current, over its whole support
                self.assertEqual(pool(config, dict(base, model='b-l', mass=mass)), 'fast')
                self.assertEqual(worker._conditional_pool.call_args.args[1:], ('b-l',))
            for fields in (dict(mass=5.1), dict(execution='reference')):
                self.assertIsNone(pool(config, dict(base, model='b-l', **fields)))
            with patch('exhad.exclusive.sample_hadronic', return_value=[]) as exclusive:  # below the start masses
                for model, mass in (('alp-fermion', 1.9), ('dark-photon', 1.69), ('scalar-1809', 1.99), ('b-l', 1.9)):
                    reference.reset_mock()
                    worker.sample(config, dict(base, model=model, mass=mass))
                    self.assertEqual((exclusive.call_args.args[1]['model'], reference.called), (model, False))
            with self.assertRaises(KeyError):  # a configuration without the accelerator is not served
                pool({}, base)
            with patch('exhad.transition.sample') as transition:  # the matched share above 4 GeV
                worker.sample(config, dict(base, mass=4.4))
                self.assertEqual(transition.call_args.args[2].keywords, {'active_pool': 'fast'})
                worker.sample(config, dict(base, mass=4.4, weighted=True))
                self.assertIs(transition.call_args.args[2], reference)
        with self.assertRaises(ValueError):
            worker.sample({}, dict(base, execution='unknown'))


    def test_cpp_hnl_sample_of_decays_with_hadrons_uses_the_mixing(self):
        """The C++ client asks for the decays that contain hadrons by sending the three squared
        mixings beside the scope; that request reaches the generator with them."""
        calls = []

        class Recorder:
            def generate(self, mass, events, **kw):
                calls.append(('generate', mass, events, kw))
                return [[0.] * 8] * events

            def generate_all(self, mass, events, **kw):
                calls.append(('generate_all', mass, events, kw))
                return [[0.] * 8] * events

        generators = {}
        request = worker.cpp_request('hnl central auto 0 512 3 2 123 hadronic 1 0 0')
        key = tuple(request[name] for name in ('model', 'variation', 'execution', 'workers', 'chunk_size'))
        generators[key] = Recorder()
        worker.cpp_sample(generators, dict(request))
        self.assertEqual(calls[-1][0], 'generate')
        self.assertEqual(calls[-1][3].get('mixing'), [1., 0., 0.])

        request = worker.cpp_request('dark-photon central auto 0 512 3 2 123 hadronic 0 0 0')
        key = tuple(request[name] for name in ('model', 'variation', 'execution', 'workers', 'chunk_size'))
        generators[key] = Recorder()
        worker.cpp_sample(generators, dict(request))
        self.assertIsNone(calls[-1][3].get('mixing'))
    def test_cpp_line_protocol(self):
        row = 'alp-fermion central auto 0 512 2 3 123 all 0 0 0'
        self.assertEqual(worker.cpp_request(row), dict(model='alp-fermion', variation='central', execution='auto',
                         workers=None, chunk_size=512, mass=2., events=3, seed=123, all_decays=True, mixing=None))
        self.assertEqual(worker.cpp_request(row.replace(' 0 512 ', ' 4 8 '))['workers'], 4)
        rows = 'hnl central auto 0 512 1 5 7 rows 1 0 0 matched 2Pie=3 Pi0v=2'
        self.assertEqual(worker.cpp_request('hnl central auto 0 512 3 1 123 hadronic 1 0 0'), dict(  # generate: hadronic rows
            model='hnl', variation='central', execution='auto', workers=None, chunk_size=512,
            mass=3., events=1, seed=123, all_decays=False, mixing=[1., 0., 0.]))
        self.assertEqual(worker.cpp_request(rows), dict(model='hnl', variation='central', execution='auto',
                         workers=None, chunk_size=512, mass=1., events=5,
                         seed=7, all_decays=False, mixing=[1., 0., 0.], rows={'2Pie': 3, 'Pi0v': 2}, terminal='matched'))
        empty = 'hnl central auto 0 512 1 0 7 rows 1 0 0 matched'  # the header client asks for no row at all
        self.assertEqual(worker.cpp_request(empty), dict(model='hnl', variation='central', execution='auto',
                         workers=None, chunk_size=512, mass=1., events=0,
                         seed=7, all_decays=False, mixing=[1., 0., 0.], rows={}, terminal='matched'))
        for invalid in (rows.replace('=3', '=4'), rows.replace('Pi0v=2', 'Pi0v=-2 2Pie=7'), rows.replace(' 2Pie=3 Pi0v=2', ''),
                        rows.replace('Pi0v=2', 'Pi0v'), row.replace('all', 'rows'), row.replace('all', 'rows') + ' pythia'):
            with self.assertRaises(ValueError):
                worker.cpp_request(invalid)
        for rows_, terminal, mixing in (([('a', 1)], 'pythia', None), ({'a': -1}, 'pythia', None), ({'a': 1}, 'full', None),
                                        ({'': 1}, 'pythia', None), ({'a': True}, 'pythia', None)):
            with self.assertRaises(ValueError):
                check_request('dark-photon', 'generate_rows', 1., 0, 1, rows=rows_, terminal=terminal, mixing=mixing)
        for model, mixing in (('hnl', None), ('dark-photon', [1, 0, 0])):
            with self.assertRaisesRegex(ValueError, 'mixing'):
                check_request(model, 'generate_rows', 1., 0, 1, rows={'a': 1}, mixing=mixing)
        for invalid in (row.replace('alp-fermion', 'unknown-model'), row.replace(' 2 ', ' nan '), row.replace(' 3 ', ' -1 '),
                        row.replace(' 123 ', ' -1 '), row.replace('all', 'unknown'), row + ' trailing',
                        row.replace(' 0 512 ', ' -1 512 '), row.replace(' 0 512 ', ' 65 512 '),
                        row.replace(' 0 512 ', ' 0 0 '), row.replace(' 0 512 ', ' 1.5 512 '),
                        'hnl central auto 0 512 3 1 123 all 0 0 0'):
            with self.assertRaises(ValueError):
                worker.cpp_request(invalid)

    def test_hnl_current_identification(self):
        # (current, conjugate): generators use hadronic charge -1, so l- plus a charge +1 pair is conjugated
        for pdgs, expected in (([11, 2, -1], ('CC_ud', True)), ([-11, -2, 1], ('CC_ud', False)),
                               ([13, 2, -3], ('CC_us', True)), ([11, -1, 4], ('CC_cd', True)),
                               ([-11, -4, 3], ('CC_cs', False)), ([11, 2, -5], ('CC_ub', True)),
                               ([12, 1, -1], ('NC_ud', False)), ([14, 4, -4], ('NC_c', False))):
            self.assertEqual(channel_spec(pdgs), expected)
        for invalid in ([-11, 2, -1], [12, 2, -1], [11, 2, -2], [11, 211, -211], [11, 12, 2, -1], [-11, 4, -3]):
            self.assertIsNone(channel_spec(invalid))

    def test_host_model_info(self):
        from exhad import model_info
        from exhad.models import MODELS
        for model in MODELS:
            info = json.loads(json.dumps(model_info(model)))  # plain data
            self.assertTrue(all(Path(path).is_file() for path in info['tables'].values()))
            if 'eventcalc_rows' in info:
                self.assertEqual(sorted(row['authority_row_id'] for row in info['eventcalc_rows']),
                                 sorted(info['owner_probabilities']['probabilities']))
        self.assertEqual(model_info('alp-fermion')['support_gev'], [1.911, 5.0])
        self.assertEqual(model_info('scalar-central')['support_gev'], [2.0, 63.0])

    def test_default_mother_pdg(self):
        """Every model has a mother PDG code; C++ exhad::defaultMotherPdg returns the same one."""
        from exhad import model_info
        from exhad.models import MODELS, MOTHER_PDG, SCALARS, mother_pdg
        self.assertEqual(sorted(MOTHER_PDG), sorted(MODELS))
        self.assertEqual({model_info(model)['mother_pdg'] for model in SCALARS}, {35})
        self.assertEqual([mother_pdg(model) for model in ('dark-photon', 'b-l', 'alp-fermion', 'hnl')],
                         [4900022, 32, 36, 9900012])
        self.assertEqual((mother_pdg('hnl', -9900012), mother_pdg('alp-fermion', 9000005)), (-9900012, 9000005))
        for invalid in (0, True, 2**31, -2**31, 36.):
            with self.assertRaises(ValueError):
                mother_pdg('alp-fermion', invalid)
        with self.assertRaises(ValueError):
            mother_pdg('unknown-model')
        source = (ROOT / 'include/exhad/Generator.hpp').read_text()
        body = source[source.index('inline int defaultMotherPdg'):]
        rules = re.findall(r'if \(model(?: == "([^"]+)"|\.rfind\("([^"]+)", 0\) == 0)\) return (\d+);',
                           body[:body.index('throw')])
        for model in MODELS:
            code = next(int(value) for exact, prefix, value in rules
                        if model == exact or prefix and model.startswith(prefix))
            self.assertEqual(code, MOTHER_PDG[model], model)


class Robustness(unittest.TestCase):
    """One HNL threshold table; non-finite or malformed replies become errors, never crashes or NaN events."""

    def test_hnl_primary_check_uses_the_router_thresholds(self):
        self.assertEqual(hnl.PRIMARY_W_MIN, {**hnl.THRESHOLDS, 'NC_c': 2.0 * m0(411)})
        for pdgs, current, light in (([11, 4, -1], 'CC_cd', m0(211)), ([11, 4, -3], 'CC_cs', m0(321))):
            threshold = hnl.THRESHOLDS[current]

            def decays(current, W_values, seed, variation):  # rest-frame D0 plus a light meson below W
                return [[0., 0., .001, 0., m0(421), 421., 0., 0., -.001, 0., light, 211.] for _ in W_values]
            for w, accepted in ((threshold + 1e-4, True), (threshold + 4.7e-3, True), (threshold - 3e-6, False)):
                mass = w + .25
                e = (mass * mass + .000511**2 - w * w) / (2 * mass)
                p = math.sqrt(e * e - .000511**2)
                row = [0., 0., p, e, .000511, 11., 0., 1.]
                for pid, sign in zip(pdgs[1:], (1, -1)):
                    row += [sign * w / 2, 0., -p / 2, (mass - e) / 2, 0., pid, 0., 0.]
                request = dict(variation='central', mass=mass, pdgs=pdgs, primary_events=[row], seed=3)
                with patch.object(hnl, 'decays_at_W', side_effect=decays):
                    if not accepted:
                        with self.assertRaisesRegex(RuntimeError, 'below the physical hadronic threshold'):
                            hnl.hadronize({}, request)
                        continue
                    (event,) = hnl.hadronize({}, request)
                hadrons = np.asarray(event[1:])
                np.testing.assert_allclose(hadrons[:, :4].sum(axis=0), np.add(row[8:12], row[16:20]), rtol=0, atol=2e-9 * w)

    def test_short_dedicated_runs_continue_with_derived_seeds(self):
        """A run that ends short is continued by fresh runs; nothing is recycled or substituted."""
        built = (ROOT / 'cpp/.pythia8-build').is_file()  # without a build (CI) a placeholder XML path suffices
        with patch.dict('os.environ', {} if built else dict(EXHAD_PYTHIA8DATA='unused')):
            from exhad.hnl_backends import dedicated
        calls = []

        def runs(yields):
            def run(count, seed):
                calls.append((count, seed))
                return [(seed, i) for i in range(min(count, yields.pop(0)))]
            return run
        self.assertEqual(dedicated._complete(runs([5]), 5, 42, 'x'), [(42, i) for i in range(5)])
        self.assertEqual(calls, [(5, 42)])  # a run that is not short is the request, unchanged
        for replay in range(2):
            calls.clear()
            events = dedicated._complete(runs([2, 0, 1, 9]), 5, 42, 'x')
            self.assertEqual([count for count, _ in calls], [5, 3, 3, 2])
            seeds = [seed for _, seed in calls]
            self.assertEqual(events, [(42, 0), (42, 1), (seeds[2], 0), (seeds[3], 0), (seeds[3], 1)])
            self.assertTrue(seeds[0] == 42 and len(set(seeds)) == 4 and all(1 <= s <= 899_900_000 for s in seeds[1:]))
            if replay:
                self.assertEqual(seeds, first)
            first = seeds
        with self.assertRaisesRegex(RuntimeError, 'closed returned 0/3 events after 1000 continuation runs'):
            dedicated._complete(lambda count, seed: [], 3, 7, 'closed')

    def test_dilation_fails_closed(self):
        out = hnl._dilate([0., 0., .3, 0., .13957, 211., 0., 0., -.3, 0., .13957, -211.], 1., final=True)
        self.assertAlmostEqual(out[3] + out[9], 1., places=12)
        nan = float('nan')
        for event, w in (([0., 0., 0., 0., .13957, 211., 0., 0., 0., 0., .13957, -211.], 1.),  # no momentum: inf * 0
                         ([0., 0., nan, 0., .13957, 211., 0., 0., .3, 0., .13957, -211.], 1.),
                         ([0., 0., .3, 0., .13957, 211., 0., 0., -.3, 0., .13957, -211.], nan),
                         ([0., 0., 0., 0., nan, 111.], 1.)):
            for final in (False, True):
                with self.assertRaises(RuntimeError):
                    hnl._dilate(event, w, final=final)

    def test_worker_replies_error_and_survives_invalid_events(self):
        good, nan, inf = [[0., 0., 1., 1., 0., 22.], [0., 0., -1., 1., 0., 22.]], float('nan'), float('inf')
        invalid = [[[0., 0., nan, 1., 0., 22.]], [[0., 0., 1., inf, 0., 22.]], [[0., 0., 1., -1., 0., 22.]],
                   [[0., 0., 1., 1., -1e-3, 22.]], [[0., 0., 1., 1., 0., 211.5]], [[0., 0., 1., 1., 0., 0.]],
                   [[0., 0., 1., 1., 0.]]]
        replies = [[event] for event in invalid] + [dict(events=[good], raw_weights=[nan]), [good]]
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / '.runtime').mkdir()
            (Path(tmp) / '.runtime/current.json').write_text('{"xmldoc": "unused", "conditional_acceleration": "unused"}')
            for cpp, line in ((False, '{"model": "alp-fermion"}\n'), (True, 'alp-fermion central auto 0 512 2 1 1 hadronic 0 0 0\n')):
                stdout = io.StringIO()
                with patch.object(worker, 'ROOT', Path(tmp)), \
                        patch.object(worker, 'cpp_sample' if cpp else 'sample', side_effect=replies), \
                        patch('sys.stdin', io.StringIO(line * len(replies))), patch('sys.stdout', stdout), \
                        patch('sys.stderr', io.StringIO()), patch.dict('os.environ'):
                    worker.main(cpp=cpp)
                lines = stdout.getvalue().splitlines()
                if cpp:
                    self.assertTrue(all(row.startswith('ERROR ValueError: ') for row in lines[:-5]))
                    self.assertEqual(lines[-5:], ['OK 1', 'EVENT 2', '0 0 1 1 0 22', '0 0 -1 1 0 22', 'END'])
                else:
                    self.assertTrue(all(json.loads(row)['error'].startswith('ValueError: ') for row in lines[:-1]))
                    self.assertEqual(json.loads(lines[-1]), {'events': [good]})
                self.assertEqual(len(lines), len(replies) + 4 * cpp)

    def test_secondary_decayer_rejects_invalid_replies(self):
        from exhad.secondary import Decayer
        replies = ['1 0 0 nan 1 0 22', '1 0 0 1 inf 0 22', '1 0 0 1 -1 0 22', '1 0 0 1 1 -0.001 22', '2 0 0 1 1 0 22',
                   '1 0 0 1 1 0 22']
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / 'decayer.py'
            script.write_text(f'import sys\nfor line, reply in zip(sys.stdin, {replies!r}):\n    print(reply, flush=True)\n')
            decayer = Decayer(sys.executable, str(script))
            try:
                for _ in replies[:-1]:
                    with self.assertRaisesRegex(RuntimeError, 'Malformed secondary-decay reply'):
                        decayer.finish([[[0., 0., 1., 1., 0., 22.]]], 1)
                self.assertEqual(decayer.finish([[[0., 0., 1., 1., 0., 22.]]], 1), [[[0., 0., 1., 1., 0., 22.]]])
            finally:
                decayer.close()


class FakeWorker:
    instances, fail = [], False

    def __init__(self, model, **options):
        self.closed = False
        self.instances.append(self)

    def generate(self, mass, n, *, seed):
        if self.fail:
            raise RuntimeError('failed chunk')
        return [(mass, seed, i) for i in range(n)]

    def generate_weighted(self, mass, n, *, seed, weight_floor_fraction):
        return dict(events=self.generate(mass, n, seed=seed), raw_weights=[float(2 + i) for i in range(n)],
                    normalization_groups=['fragmentation'] * n, family_labels=['test'] * n,
                    weight_convention='self-normalized-within-fragmentation; external-unit-weight',
                    weight_floor_fraction=weight_floor_fraction)

    def hadronize_hnl(self, mass, pdgs, primary_events, *, seed):
        return [(seed, row) for row in primary_events]

    def close(self):
        self.closed = True


class ParallelScheduling(unittest.TestCase):
    def setUp(self):
        FakeWorker.instances, FakeWorker.fail = [], False
        patcher = patch('exhad.api.Worker', FakeWorker)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_worker_count_independence_and_raw_weights(self):
        with Generator('alp-fermion', workers=1, chunk_size=5) as a, \
                Generator('alp-fermion', workers=4, chunk_size=5) as b, Generator('alp-fermion', chunk_size=5) as c:
            for n in (0, 1, 23):
                x = a.generate(3., n, seed=2)
                self.assertEqual((x, len(set(x))), (b.generate(3., n, seed=2), n))
                self.assertEqual(x, c.generate(3., n, seed=2))
            self.assertEqual(a.generate(3., 5, seed=2), [(3., 2, i) for i in range(5)])  # one chunk: the request seed
            b.generate(2., 13, seed=1)
            self.assertEqual(x, b.generate(3., 23, seed=2))
            weighted = a.generate_weighted(3., 13, seed=8, weight_floor_fraction=.1)
            self.assertEqual(weighted, b.generate_weighted(3., 13, seed=8, weight_floor_fraction=.1))
            self.assertEqual(weighted['raw_weights'], [2., 3., 4., 5., 6.] * 2 + [2., 3., 4.])
        self.assertTrue(all(g.closed for g in FakeWorker.instances))
        rows = [[i] * 24 for i in range(13)]
        with Generator('hnl', workers=3, chunk_size=5) as g:
            self.assertEqual([row for _, row in g.hadronize_hnl(3., [11, 2, -1], rows, seed=9)], rows)
            with self.assertRaises(ValueError):
                g.generate(3., 13)

    def test_failure_and_invalid_inputs(self):
        g = Generator('alp-fermion', workers=4, chunk_size=5)
        g.generate(3., 2)
        self.assertEqual(len(FakeWorker.instances), 1)
        FakeWorker.fail = True
        with self.assertRaisesRegex(RuntimeError, 'failed chunk'):
            g.generate(3., 23)
        FakeWorker.fail = False  # a failed request leaves every worker idle and usable
        self.assertEqual(len(g.generate(3., 23)), 23)
        g.close()
        self.assertTrue(all(w.closed for w in FakeWorker.instances))
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            g.generate(3., 1)
        FakeWorker.instances = []
        for options in (dict(workers=True), dict(workers=0), dict(workers=-1), dict(workers=1.5), dict(workers=65),
                        dict(chunk_size=0), dict(execution='fast')):
            with self.assertRaises(ValueError):
                Generator('alp-fermion', **options)
        self.assertTrue(1 <= Generator('alp-fermion').workers <= 8)
        with Generator('alp-fermion') as g:
            for n, seed in ((True, 1), (-1, 1), (1, True), (1, -1), (1, 2**64)):
                with self.assertRaises(ValueError):
                    g.generate(3., n, seed=seed)
            for floor in (True, -.1, float('nan'), 1.1):
                with self.assertRaises(ValueError):
                    g.generate_weighted(3., 1, weight_floor_fraction=floor)
        self.assertFalse(FakeWorker.instances)


class Output(unittest.TestCase):
    payload = dict(model='alp-fermion', mass_gev=2., hadronization_weights=[.5, 1.5], events=[
        [[0., 0., 1., 1., 0., 22], [0., 0., -1., 1., 0., 22]], [[1., 0., 0., 1., 0., 22], [-1., 0., 0., 1., 0., 22]]])

    def test_json_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'events.json'
            output.write_events(path, self.payload)
            original = path.read_bytes()
            self.assertEqual(json.loads(original), self.payload)
            with self.assertRaises(FileExistsError):
                output.write_events(path, {})
            self.assertEqual(path.read_bytes(), original)

    def test_hepmc_metadata_rejected_before_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'events.hepmc'
            for pid in (0, True, 2**31):
                with self.assertRaises(ValueError):
                    output.write_events(path, self.payload, format='hepmc', mother_pdg=pid)
                self.assertFalse(path.exists())

    def test_hepmc_roundtrip(self):
        import pyhepmc
        for format, pid, expected in (('hepmc', 9000006, 9000006), ('hepmc2', -9000006, -9000006), ('hepmc', None, 36)):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'events.hepmc'
                output.write_events(path, self.payload, format=format, mother_pdg=pid)
                with pyhepmc.open(path) as reader:
                    events = list(reader)
                self.assertEqual(len(events), 2)
                for i, event in enumerate(events):
                    self.assertEqual((event.event_number, event.weights), (i, [self.payload['hadronization_weights'][i]]))
                    (vertex,) = event.vertices
                    (mother,) = vertex.particles_in
                    self.assertEqual((mother.pid, mother.status, tuple(mother.momentum)), (expected, 2, (0., 0., 0., 2.)))
                    self.assertEqual([(d.pid, d.status, tuple(d.momentum)) for d in vertex.particles_out],
                                     [(22, 1, tuple(p[:4])) for p in self.payload['events'][i]])


if __name__ == '__main__':
    unittest.main()
