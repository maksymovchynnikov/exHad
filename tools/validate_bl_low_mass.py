#!/usr/bin/env python3
"""Validate data/b-l/eventcalc_* against both table loaders, ReD-DeLiVeR and exHad's B-L width (build time only).

(1) format: exHad's loader (full_decays.table/interpolate, kinematics.matrix_element) and EventCalc's
    setup_br_interpolators/compile_matrix_elements (their source text, taken from funcs/initLLP.py
    without importing EventCalc) parse labels, PDGs, rates and |M|^2 identically;
(2) native rates against ReD-DeLiVeR BL_model_tutorial_hadronic_normwid.txt at its own nodes;
(3) sum of rows against the DeLiVeR total hadronic width below the matching mass;
(4) thresholds; |M|^2(3pi) >= 0 over the Dalitz region at every table mass, equal to the Bose-symmetric
    4 m^2 |p+ x p-|^2 |sum of the three rho propagators|^2 built from exHad's generated momenta, and sampled by
    exHad with the weighted-Dalitz energy shares;
(5) the hadronic width against exHad's B-L width at the matching mass and at 2 GeV.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT)]

import numpy as np  # noqa: E402

import build_bl_low_mass as builder  # noqa: E402

# Both inputs are checkouts of other programs, which this build-time script reads but does not ship.
# Set EXHAD_EVENTCALC_DIR and EXHAD_RED_DELIVER_DIR, or place the checkouts at these default paths.
EVENTCALC = Path(os.environ.get('EXHAD_EVENTCALC_DIR') or ROOT / 'external' / 'eventcalc')
RED_DELIVER = Path(os.environ.get('EXHAD_RED_DELIVER_DIR') or ROOT / 'external' / 'red-deliver')
REFERENCE = RED_DELIVER / 'models' / 'BL_model_tutorial' / 'BL_model_tutorial_hadronic_normwid.txt'


def require_inputs():
    """Fail with the repository and the variable to set, instead of a bare missing-file error."""
    missing = []
    if not (EVENTCALC / 'funcs' / 'initLLP.py').exists():
        missing.append(f'EventCalc (maksymovchynnikov/EventCalc-SHiP): EXHAD_EVENTCALC_DIR={EVENTCALC}')
    if not REFERENCE.exists():
        missing.append(f'ReD-DeLiVeR (anafoguel/ReD-DeLiVeR): EXHAD_RED_DELIVER_DIR={RED_DELIVER}')
    if missing:
        raise SystemExit('missing build-time input checkouts:\n  ' + '\n  '.join(missing))


def eventcalc_loader(initllp: Path):
    """EventCalc's two table methods, compiled from their source text into a stub object."""
    import sympy as sp
    from scipy.interpolate import RegularGridInterpolator
    tree = ast.parse(initllp.read_text())
    wanted = {'setup_br_interpolators', 'compile_matrix_elements'}
    methods = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in wanted]

    if {m.name for m in methods} != wanted:
        raise RuntimeError('EventCalc loader methods not found in ' + str(initllp))
    module = ast.Module(body=[ast.ClassDef(name='Loader', bases=[], keywords=[], body=methods, decorator_list=[])],
                        type_ignores=[])
    namespace = {'np': np, 'sp': sp, 're': re, 'RegularGridInterpolator': RegularGridInterpolator}
    exec(compile(ast.fix_missing_locations(module), str(initllp), 'exec'), namespace)
    loader = namespace['Loader']()
    loader.Matrix_elements_expr = []

    return loader


def eventcalc_read(path: Path, loader):
    """EventCalc's DP-style read: pandas.read_json, columns 0, 1, 2 and -1."""
    import pandas as pd
    frame = pd.read_json(path)
    channels = frame.iloc[:, 0].to_numpy()
    pdgs = frame.iloc[:, 1].apply(np.array).to_numpy()
    rates = frame.iloc[:, 2].to_numpy()
    expressions = frame.iloc[:, -1].to_numpy()

    return channels, pdgs, loader.setup_br_interpolators(rates), loader.compile_matrix_elements(expressions)


def dalitz_points(mass, m1, m2, m3, n=41):
    """Regular (u, v) grid over the Dalitz region, boundary included: E1 (particle 1) and E3 (particle 3)."""
    u = np.linspace(0.0, 1.0, n)
    s12 = (m1 + m2)**2 + u[:, None] * ((mass - m3)**2 - (m1 + m2)**2)
    s12 = np.broadcast_to(s12, (n, n))
    e2s = (s12 - m1**2 + m2**2) / (2 * np.sqrt(s12))       # particle 2 energy in the 12 frame
    e3s = (mass**2 - s12 - m3**2) / (2 * np.sqrt(s12))      # particle 3 energy in the 12 frame
    p2, p3 = np.sqrt(np.maximum(e2s**2 - m2**2, 0)), np.sqrt(np.maximum(e3s**2 - m3**2, 0))
    lo, hi = (e2s + e3s)**2 - (p2 + p3)**2, (e2s + e3s)**2 - (p2 - p3)**2
    s23 = lo + u[None, :] * (hi - lo)
    e3 = (mass**2 + m3**2 - s12) / (2 * mass)
    e1 = (mass**2 + m1**2 - s23) / (2 * mass)

    return e1.ravel(), e3.ravel()


def check_format(report, table_path, labels_expected=None):
    from exhad.full_decays import interpolate, table
    from exhad.core.kinematics import PARTICLES, matrix_element
    rows = table(str(table_path))
    loader = eventcalc_loader(EVENTCALC / 'funcs' / 'initLLP.py')
    channels, pdgs, get_br, compiled = eventcalc_read(table_path, loader)
    out = {'rows': len(rows)}
    assert all(isinstance(r, list) and len(r) == 4 and isinstance(r[0], str) and isinstance(r[3], str) for r in rows)
    out['labels_identical'] = [r[0] for r in rows] == list(channels)
    out['pdgs_identical'] = all([int(p) for p in r[1]] == [int(p) for p in q] for r, q in zip(rows, pdgs))
    grid = np.array([p[0] for p in rows[0][2]])
    out['common_grid'] = all(np.array_equal(np.array([p[0] for p in r[2]]), grid) for r in rows)
    out['grid_strictly_increasing'] = bool(np.all(np.diff(grid) > 0))
    rng = np.random.default_rng(5)
    probe = np.unique(np.concatenate([grid, 0.5 * (grid[1:] + grid[:-1]), rng.uniform(grid[0], grid[-1], 3000)]))
    worst = 0.0
    sub = probe[::5]
    theirs_all = np.array([get_br(m) for m in sub])            # EventCalc: every row at each mass

    for i, r in enumerate(rows):
        values = np.array(r[2], dtype=float)
        mine = np.interp(sub, values[:, 0], values[:, 1])      # the arithmetic of exHad's interpolate
        spot = np.array([interpolate(r[2], m) for m in sub[::25]])  # exHad's interpolate itself
        scale = max(values[:, 1].max(), 1e-300)
        worst = max(worst, float(np.max(np.abs(theirs_all[:, i] - mine)) / scale),
                    float(np.max(np.abs(spot - theirs_all[::25, i])) / scale))
    out['rate_max_abs_diff_over_row_max'] = worst
    me_worst, me_rows = 0.0, 0

    for i, r in enumerate(rows):
        ids = [int(p) for p in r[1] if int(p) != -999]
        if r[3].strip() in {'1.', '1.0'} or len(ids) != 3:
            continue
        me_rows += 1
        masses = tuple(PARTICLES[p][0] for p in ids)
        f_exhad = matrix_element(r[3], masses)
        for mass in np.linspace(sum(masses) + 1e-3, grid[-1], 25):
            e1, e3 = dalitz_points(mass, *masses, n=21)
            a = np.asarray(f_exhad(mass, e1, e3), dtype=float)
            b = np.asarray(compiled[i](mass, e1, e3), dtype=float)
            me_worst = max(me_worst, float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300)))
    out['three_body_matrix_elements_compared'] = me_rows
    out['matrix_element_max_rel_diff'] = me_worst
    out['eventcalc_expression_strings'] = sum(e != '1.0' for e in loader.Matrix_elements_expr)

    if labels_expected is not None:
        out['labels_equal_dark_photon_labels_and_pdgs'] = labels_expected
    report.update(out)

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--processes', type=int, default=8)
    args = parser.parse_args()
    require_inputs()
    from exhad.b_l import rates
    from exhad.core.kinematics import PARTICLES, matrix_element
    meta = json.loads(builder.META.read_text())
    report = {}

    # (1) format -----------------------------------------------------------------------------
    dp = builder.dark_photon_rows()
    fmt = {}
    rows = check_format(fmt, builder.TABLE)
    fmt['pdgs_and_placeholders_equal_dark_photon_rows'] = all(r[1] == dp[r[0]][1] for r in rows)
    fmt['unit_matrix_elements_equal_dark_photon_rows'] = all(r[3] == dp[r[0]][3] for r in rows if r[0] != 'Pip_Pim_Pi0')
    fmt['compact_json_like_dark_photon_file'] = builder.TABLE.read_bytes()[:3] == builder.DP_TABLE.read_bytes()[:3]
    widths = np.loadtxt(builder.WIDTHS)
    grid = np.array([p[0] for p in rows[0][2]])
    table = np.array([[p[1] for p in r[2]] for r in rows])
    fmt['width_file_grid_equal'] = bool(np.array_equal(widths[:, 0], grid))
    fmt['width_file_equals_sum_of_rows_max_rel'] = float(np.max(np.abs(widths[:, 1] - table.sum(0)) / np.maximum(widths[:, 1], 1e-300)))
    dp_fmt = {}
    check_format(dp_fmt, builder.DP_TABLE)
    fmt['same_harness_on_dark_photon_table'] = {k: dp_fmt[k] for k in ('rows', 'labels_identical', 'pdgs_identical',
                                                                         'rate_max_abs_diff_over_row_max',
                                                                         'matrix_element_max_rel_diff')}
    report['1_format'] = fmt

    # (2) ReD-DeLiVeR ---------------------------------------------------------------------
    head = REFERENCE.read_text().splitlines()[0].split('\t')
    ref = np.loadtxt(REFERENCE, skiprows=1)
    evaluated = builder.evaluate_masses(ref[:, 0], args.processes)
    col = {name: head.index(name) for name in head}
    red = {}
    for name, subs in builder.REFERENCE_COLUMNS.items():
        mine = np.array([math.fsum(w[s] for s in subs) for w, _ in evaluated])
        theirs = ref[:, col[name]]
        both = (theirs > 0) | (mine > 0)
        rel = np.where(theirs > 0, np.abs(mine - theirs) / np.where(theirs > 0, theirs, 1), np.inf)[both]
        # table (threshold rule + interpolation) at the reference masses, away from generator thresholds
        labels = [label for label, splits in builder.ROWS if all(s in subs for s, _ in splits)]
        tab = sum(np.interp(ref[:, 0], grid, table[[r[0] for r in rows].index(l)]) for l in labels)
        th = max(meta['rows'][l]['threshold_gev'] for l in labels)
        away = (ref[:, 0] > th + 2e-3) & (theirs > 0)
        red[name] = {'nodes_compared': int(both.sum()), 'max_rel_native': float(rel.max()),
                     'at_mass': float(ref[:, 0][both][int(np.argmax(rel))]),
                     'max_rel_table_interpolated': float(np.max(np.abs(tab[away] - theirs[away]) / theirs[away])),
                     'max_abs_table_minus_ref_over_tothad_plus_m_over_12pi': float(np.max(
                         np.abs(tab[away] - theirs[away]) / (ref[away, col['tothad']] + ref[away, 0] / (12 * math.pi))))}
    iso_ref = [name for name in builder.ISOVECTOR if name in col]
    red['isovector_reference_columns_all_zero'] = bool(np.all(ref[:, [col[n] for n in iso_ref]] == 0))
    red['isovector_native_all_zero'] = all(v == 0 for _, iso in evaluated for v in iso.values())
    report['2_redeliver'] = red

    # (3) sum of rows vs DeLiVeR total below m_x ------------------------------------------------
    mx = meta['matching_mass_gev']
    below = grid < mx
    native = builder.evaluate_masses(grid[below], args.processes)
    total_native = np.array([math.fsum(w.values()) + math.fsum(iso.values()) for w, iso in native])
    rowsum = table.sum(0)[below]
    diff = rowsum - total_native
    pos = total_native > 0
    ref_tot = ref[:, col['tothad']]
    ref_rows = np.interp(ref[:, 0], grid, table.sum(0))
    sel = (ref[:, 0] < mx) & (ref_tot > 0)
    last_threshold = max(v['threshold_gev'] for v in meta['rows'].values() if v['threshold_gev'] < mx)
    report['3_sum_vs_deliver_total'] = {
        'grid_nodes_below_mx': int(below.sum()),
        'max_rel_sum_rows_minus_native_total': float(np.max(np.abs(diff[pos]) / total_native[pos])),
        'max_rel_excluding_nodes_within_1MeV_above_a_threshold': float(np.max(
            (np.abs(diff) / np.where(pos, total_native, 1))[pos & np.array([
                all(not (t < m <= t + 1e-3) for t in meta['grid']['threshold_nodes_gev']) for m in grid[below]])])),
        'removed_by_threshold_rule_max_abs_per_g2': float(-diff.min()) if diff.min() < 0 else 0.0,
        'table_sum_vs_redeliver_tothad_max_rel': float(np.max(np.abs(ref_rows[sel] - ref_tot[sel]) / ref_tot[sel])),
        'table_sum_vs_redeliver_tothad_max_rel_m_gt_1.05': float(np.max(
            np.abs(ref_rows[sel & (ref[:, 0] > 1.05)] - ref_tot[sel & (ref[:, 0] > 1.05)]) / ref_tot[sel & (ref[:, 0] > 1.05)])),
        'last_threshold_below_mx': last_threshold,
    }

    # (4) thresholds and 3pi |M|^2 ------------------------------------------------------------
    th = {}
    bad = []
    for r in rows:
        t = meta['rows'][r[0]]['threshold_gev']
        values = np.array(r[2])
        if not np.any(values[:, 0] == t):
            bad.append((r[0], 'threshold not a node'))
        if np.any(values[values[:, 0] <= t, 1] != 0):
            bad.append((r[0], 'nonzero at or below threshold'))
        if np.any(values[:, 1] < 0):
            bad.append((r[0], 'negative'))
        from exhad.full_decays import interpolate
        eps = 1e-6
        first = values[values[:, 0] > t][0]
        expect = first[1] * eps / (first[0] - t) if first[0] - t > eps else float('nan')
        got = interpolate(r[2], t + eps)
        if interpolate(r[2], t) != 0 or interpolate(r[2], t - eps) != 0 or not math.isclose(got, expect, rel_tol=1e-6, abs_tol=1e-300):
            bad.append((r[0], 'interpolation not linear from threshold'))
        # exHad's threshold-aware interpolation (threshold = sum of generator masses) gives the same values
        for x in (t - eps, t, t + eps, t + 0.3 * (first[0] - t), first[0], 0.5 * (t + values[-1, 0]), values[-1, 0]):
            if x < values[0, 0]:
                continue
            if not math.isclose(interpolate(r[2], x, threshold=t), interpolate(r[2], x), rel_tol=1e-12, abs_tol=1e-300):
                bad.append((r[0], f'threshold-aware interpolation differs at {x}'))
    th['violations'] = bad
    th['first_positive_node_above_threshold_gev'] = {
        r[0]: float(np.array(r[2])[np.array(r[2])[:, 1] > 0][0, 0]) for r in rows}
    ids = (211, -211, 111)
    masses = tuple(PARTICLES[p][0] for p in ids)
    expr = next(r[3] for r in rows if r[0] == 'Pip_Pim_Pi0')
    f_new = matrix_element(expr, masses)
    f_dp = matrix_element(dp['Pip_Pim_Pi0'][3], masses)
    worst_neg, n_points, dp_negative_share = 0.0, 0, {}
    for mass in grid[grid > sum(masses)]:
        e1, e3 = dalitz_points(mass, *masses, n=41)
        v = np.asarray(f_new(mass, e1, e3), dtype=float)
        n_points += v.size
        worst_neg = min(worst_neg, float(v.min() / v.max()))
        if any(abs(mass - x) < 5e-5 for x in (0.43, 0.6, 0.782, 1.0, 1.5)):
            w = np.asarray(f_dp(mass, e1, e3), dtype=float)
            dp_negative_share[f'{mass:.3f}'] = float(-w[w < 0].sum() / np.abs(w).sum())
    # Gram factor, and the whole |M|^2, against explicit momenta from exHad's flat three-body generator:
    # 4 m^2 |p+ x p-|^2 |sum over pion pairs of 1/(M^2 - s_ij - i M Gamma)|^2 with s_ij from the four-momenta.
    from exhad.core import kinematics as k
    k.seed(3)
    gram_rel, explicit_rel = 0.0, 0.0
    rho_m, rho_w = meta['three_pion_matrix_element']['rho_mass_gev'], meta['three_pion_matrix_element']['rho_width_gev']
    for mass in (0.45, 0.782, 1.019, 1.5, 1.99):
        ev = k.three_body(mass, 2000, ids, masses, tuple(PARTICLES[p][1] for p in ids),
                          tuple(PARTICLES[p][2] for p in ids), '1.').reshape(2000, 3, 8)
        p1, p2, p3 = ev[:, 0, :3], ev[:, 1, :3], ev[:, 2, :3]
        e1, e2_gen, e3 = ev[:, 0, 3], ev[:, 1, 3], ev[:, 2, 3]
        cross2 = np.sum(np.cross(p1, p2)**2, axis=1)
        e2 = mass - e1 - e3
        q1, q2, q3 = e1**2 - masses[0]**2, e2**2 - masses[1]**2, e3**2 - masses[2]**2
        gram = q1 * q2 - ((q3 - q1 - q2) / 2)**2
        gram_rel = max(gram_rel, float(np.max(np.abs(gram - cross2)) / np.max(cross2)))

        def s_pair(ea, pa, eb, pb):
            return (ea + eb)**2 - np.sum((pa + pb)**2, axis=1)
        bw = sum(1 / (rho_m**2 - s - 1j * rho_m * rho_w)
                 for s in (s_pair(e2_gen, p2, e3, p3), s_pair(e1, p1, e3, p3), s_pair(e1, p1, e2_gen, p2)))
        explicit = 4 * mass**2 * cross2 * np.abs(bw)**2
        table_value = np.asarray(f_new(mass, e1, e3), dtype=float)
        explicit_rel = max(explicit_rel, float(np.max(np.abs(table_value - explicit)) / np.max(explicit)))
    # Bose symmetry at the generator masses: pi+ <-> pi- (E1 <-> E2) exactly, since m(pi+) = m(pi-)
    rng = np.random.default_rng(17)
    swap_rel, dp_swap_rel = 0.0, 0.0
    for mass in (0.45, 0.782, 1.019, 1.5, 1.99):
        e1 = rng.uniform(masses[0], mass / 2, 20000)
        e3 = rng.uniform(masses[2], mass / 2, 20000)
        e2 = mass - e1 - e3
        q1, q2, q3 = e1**2 - masses[0]**2, e2**2 - masses[1]**2, e3**2 - masses[2]**2
        inside = (q1 > 0) & (q2 > 0) & (q3 > 0) & (q1 * q2 - ((q3 - q1 - q2) / 2)**2 > 0)
        e1, e2, e3 = e1[inside], e2[inside], e3[inside]
        for fn, key in ((f_new, 'table'), (f_dp, 'dp')):
            a = np.asarray(fn(mass, e1, e3), dtype=float)
            b = np.asarray(fn(mass, e2, e3), dtype=float)
            rel = float(np.max(np.abs(a - b)) / np.max(np.abs(a)))
            if key == 'table':
                swap_rel = max(swap_rel, rel)
            else:
                dp_swap_rel = max(dp_swap_rel, rel)
    # exHad samples the table |M|^2: mean energy shares of (pi+, pi-, pi0) against the weighted Dalitz average
    shares = {}
    for mass in (0.782, 1.019, 1.5):
        n_ev = 40000
        ev = k.three_body(mass, n_ev, ids, masses, tuple(PARTICLES[p][1] for p in ids),
                          tuple(PARTICLES[p][2] for p in ids), expr).reshape(n_ev, 3, 8)
        sampled = ev[:, :, 3] / mass
        e1 = rng.uniform(masses[0], mass / 2, 400000)
        e3 = rng.uniform(masses[2], mass / 2, 400000)
        e2 = mass - e1 - e3
        q1, q2, q3 = e1**2 - masses[0]**2, e2**2 - masses[1]**2, e3**2 - masses[2]**2
        inside = (q1 > 0) & (q2 > 0) & (q3 > 0) & (q1 * q2 - ((q3 - q1 - q2) / 2)**2 > 0)
        e1, e2, e3 = e1[inside], e2[inside], e3[inside]
        weight = np.asarray(f_new(mass, e1, e3), dtype=float)
        expected = [float(np.sum(weight * e) / np.sum(weight) / mass) for e in (e1, e2, e3)]
        mean = sampled.mean(0)
        error = sampled.std(0) / math.sqrt(n_ev)
        shares[f'{mass:.3f}'] = {'sampled_pip_pim_pi0': [float(x) for x in mean], 'weighted_dalitz_pip_pim_pi0': expected,
                                 'max_pull': float(np.max(np.abs(mean - expected) / error))}
    th['three_pion_matrix_element'] = {
        'table_masses': int(np.sum(grid > sum(masses))), 'dalitz_points': n_points,
        'min_over_max_per_mass_worst': worst_neg,
        'dark_photon_string_negative_weight_share_at_exhad_masses': dp_negative_share,
        'gram_factor_vs_explicit_cross_product_max_rel': gram_rel,
        'table_string_vs_explicit_momenta_all_plus_rho_sum_max_rel': explicit_rel,
        'pip_pim_exchange_symmetry_max_rel_table': swap_rel,
        'pip_pim_exchange_symmetry_max_rel_dark_photon_string': dp_swap_rel,
        'builder_family_checks': meta['three_pion_matrix_element']['checks'],
        'exhad_sampled_energy_shares': shares,
    }
    report['4_thresholds_and_3pi'] = th

    # (5) continuity with exHad's B-L width --------------------------------------------------
    total = table.sum(0)

    def at(m):
        return float(np.interp(m, grid, total))
    w2 = rates.total_width(2.0)
    ext = rates.external_widths(2.0)
    idx2 = int(np.where(grid == 2.0)[0][0])
    row2 = {r[0]: r[2][idx2][1] for r in rows}
    known = rates.known_widths(2.0)
    report['5_continuity'] = {
        'matching_mass_gev': mx,
        'matching_mass_is_grid_node': bool(np.any(grid == mx)),
        'sum_rows_table_at_mx_over_total_width_minus_1': at(mx) / rates.total_width(mx) - 1,
        'slope_sum_rows_at_mx_per_gev': (at(mx + 5e-3) - at(mx - 5e-3)) / 1e-2,
        'slope_total_width_at_mx_per_gev': (rates.total_width(mx + 5e-3) - rates.total_width(mx - 5e-3)) / 1e-2,
        'total_width_2gev_per_g2': w2,
        'sum_rows_2gev_per_g2': float(total[idx2]),
        'jump_sum_rows_vs_exhad_width_at_2gev': float(total[idx2] / w2 - 1),
        'external_rows_2gev_vs_rates_external_widths_max_rel': max(
            abs(v / ext[k_] - 1) for k_, v in {
                'pi0-gamma-block': row2['Pi0_gamma'], 'eta-gamma-block': row2['EtaGamma'],
                'kaon-pair-block': row2['Kp_Km'] + row2['KL_KS'], 'proton-pair-block': row2['ppbar'],
                'neutron-pair-block': row2['nnbar']}.items()),
        'exact_3pi_2gev_over_native_3pi': known['exact-3pi'] / row2['Pip_Pim_Pi0'],
        'exact_kkpi_2gev_over_native_kkpi': known['exact-kkpi'] / sum(row2[l] for l in builder.KAPPA_ROWS['kkpi']),
        'kappa_three_pion': meta['kappa']['three_pion']['value'], 'kappa_kkpi': meta['kappa']['kkpi']['value'],
        'residual_share_of_total_width_unscaled': {f'{m:.2f}': 1 - at(m) / rates.total_width(m) for m in (1.8, 1.9, 2.0)},
    }
    print(json.dumps(report, indent=1, default=float))


if __name__ == '__main__':
    main()
