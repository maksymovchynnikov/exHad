#!/usr/bin/env python3
"""Build the B-L low-mass exclusive decay table (EventCalc row format) from native DeLiVeR.

Build time only; nothing under exhad/ imports this file.

Physics input: the pinned DeLiVeR form factors in external/deliver, reset to the
B-L quark charges q_u = q_d = q_s = 1/3 with unit coupling, the convention of
exhad/b_l/rates.py and ReD-DeLiVeR's BL_model_tutorial: L = g_BL Z'_mu J_{B-L}^mu,
so every width is Gamma/g_BL^2 in GeV (Gamma(Z' -> e+e-)/g^2 = m/(12 pi) for m >> m_e).
Only the isoscalar current couples: the isovector DeLiVeR channels (2pi, 4pi, 6pi,
eta pi pi, eta' pi pi, omega pi, phi pi) are evaluated and must vanish identically.

Outputs (data/b-l/):
- eventcalc_branching_ratios.json: rows [label, PDGs, [[m, Gamma/g^2], ...], |M|^2],
  labels, PDG lists (with their -999 placeholders), charge-conjugate and K_S/K_L
  splits copied from data/dark-photon/eventcalc_branching_ratios.json (EventCalc's
  DP-decay.json). The node values are partial widths per g_BL^2, not branching ratios.
- eventcalc_hadronic_widths.dat: mass and the sum of the rows, Gamma_had/g_BL^2, on the same grid.
- eventcalc_low_mass.json: grid, thresholds, matching mass m_x, the kappa factors,
  source and output sha256.

Threshold rule: a row is zero at and below the sum of exHad's generator masses
(exhad.core.kinematics.PARTICLES) and each threshold is a grid node, so plain linear
interpolation (EventCalc's RegularGridInterpolator, numpy.interp) rises linearly from
the threshold to the first node above it.

3pi |M|^2: EventCalc's dark-photon row family (fixed rho line shape, Gram factor, EventCalc variables)
at the generator pion masses, with the three rho propagators added with the same sign. For the I = 0
current (omega/phi -> rho pi, hep-ph/0512180, the model behind DeLiVeR's 3pi rates) the pion isospin
factor and the epsilon-tensor factor are both totally antisymmetric, so the form factor must be totally
symmetric: BW(s_-0) + BW(s_+0) + BW(s_+-). The dark-photon table's Pip_Pim_Pi0 row carries the same
expression, which the build checks.

SciPy 1.15 removed scipy.integrate.quadrature, which DeLiVeR's omega pi pi, phi pi pi
and K K pi pi phase-space integrals call; _quadrature below is a port of the removed
SciPy routine (same algorithm and stopping rule) and is installed only when missing.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from multiprocessing import get_context
from pathlib import Path
import platform
import subprocess
import sys
import warnings

sys.dont_write_bytecode = True  # never write __pycache__ into the pinned external checkouts

ROOT = Path(__file__).resolve().parents[1]
DELIVER = ROOT / 'external' / 'deliver'
DATA = ROOT / 'data'
OUT = DATA / 'b-l'
TABLE = OUT / 'eventcalc_branching_ratios.json'
WIDTHS = OUT / 'eventcalc_hadronic_widths.dat'
META = OUT / 'eventcalc_low_mass.json'
DP_TABLE = DATA / 'dark-photon' / 'eventcalc_branching_ratios.json'
DELIVER_REVISION = '7b2bbd79fbacfed9a01d24e903762b4759b9e4a2'

CHARGES = (1 / 3, 1 / 3, 1 / 3)  # (u, d, s) B-L charges
TABLE_START, TABLE_END = 0.001, 2.0
BASE_STEP_MEV = 1          # 1-MeV base grid, EventCalc style
# Adaptive refinement: an interval is halved while, at its midpoint, linear interpolation of any row or of the
# row sum misses the native width by more than REFINE_TOL x (Gamma_had + m/(12 pi)), i.e. 1e-4 of the width scale
# including the unit leptonic width, so branching fractions are interpolated to about 1e-4. Smallest step 1/64 MeV.
REFINE_TOL = 1e-4
REFINE_PASSES = 6
MATCH_BRACKET = (1.70, 1.80)

# DeLiVeR submodes of the isoscalar current: name -> (module, function, mode).
SUBMODES = {
    '3pi': ('F3pi', 'GammaDM', None),
    'PiGamma': ('FPiGamma', 'GammaDM', None),
    'EtaGamma': ('FEtaGamma', 'GammaDM', None),
    'EtaOmega': ('FEtaOmega', 'GammaDM', None),
    'EtaPhi': ('FEtaPhi', 'GammaDM', None),
    'KK_n': ('FK', 'GammaDM_mode', 0),
    'KK_c': ('FK', 'GammaDM_mode', 1),
    'KKpi_0': ('FKKpi', 'GammaDM_mode', 0),
    'KKpi_1': ('FKKpi', 'GammaDM_mode', 1),
    'KKpi_2': ('FKKpi', 'GammaDM_mode', 2),
    'KKpipi_0': ('FKKpipi', 'GammaDM_mode', 0),
    'KKpipi_1': ('FKKpipi', 'GammaDM_mode', 1),
    'KKpipi_2': ('FKKpipi', 'GammaDM_mode', 2),
    'KKpipi_3': ('FKKpipi', 'GammaDM_mode', 3),
    'PhiPiPi_n': ('FPhiPiPi', 'GammaDM_mode', 0),
    'PhiPiPi_c': ('FPhiPiPi', 'GammaDM_mode', 1),
    'OmPiPi_n': ('FOmPiPi', 'GammaDM_mode', 0),
    'OmPiPi_c': ('FOmPiPi', 'GammaDM_mode', 1),
    'ppbar': ('Fppbar', 'GammaPDM', None),
    'nnbar': ('Fppbar', 'GammaNDM', None),
}
# Isovector DeLiVeR channels: the B-L current does not couple (q_u - q_d = 0); checked to be exactly zero.
ISOVECTOR = {
    '2pi': ('F2pi', 'GammaDM', None), '4pi': ('F4pi', 'GammaDM', None), '6pi': ('F6pi', 'GammaDM', None),
    'EtaPiPi': ('FEtaPiPi', 'GammaDM', None), 'EtaPrimePiPi': ('FEtaPrimePiPi', 'GammaDM', None),
    'OmegaPion': ('FOmegaPion', 'GammaDM', None), 'PhiPi': ('FPhiPi', 'GammaDM', None),
}
# ReD-DeLiVeR hadronic_normwid column -> submodes (its aggregated columns).
REFERENCE_COLUMNS = {
    '3pi': ('3pi',), 'PiGamma': ('PiGamma',), 'EtaGamma': ('EtaGamma',), 'EtaOmega': ('EtaOmega',),
    'EtaPhi': ('EtaPhi',), 'KK': ('KK_n', 'KK_c'), 'KKpi': ('KKpi_0', 'KKpi_1', 'KKpi_2'),
    'KKpipi': ('KKpipi_0', 'KKpipi_1', 'KKpipi_2', 'KKpipi_3'), 'PhiPiPi': ('PhiPiPi_n', 'PhiPiPi_c'),
    'OmPiPi': ('OmPiPi_n', 'OmPiPi_c'), 'NNbar': ('ppbar', 'nnbar'),
}

_CONJUGATE_SPLIT = (('KS_plus', 0.25), ('KS_minus', 0.25), ('KL_plus', 0.25), ('KL_minus', 0.25))
# (row label, ((submode, fraction), ...)); order and splits follow the dark-photon EventCalc table.
ROWS = (
    ('KL_KS', (('KK_n', 1.0),)),
    ('Kp_Km', (('KK_c', 1.0),)),
    ('KKpi_0_KL_KS_Pi0', (('KKpi_0', 1.0),)),
    ('Kp_Km_Pi0', (('KKpi_1', 1.0),)),
    *((f'KKpi_2_{suffix}', (('KKpi_2', f),)) for suffix, f in _CONJUGATE_SPLIT),
    ('EtaPhi', (('EtaPhi', 1.0),)),
    ('KKpipi_0', (('KKpipi_0', 1.0),)),
    *((f'KKpipi_{k}_{suffix}', ((f'KKpipi_{k}', f),)) for k in (1, 2, 3) for suffix, f in _CONJUGATE_SPLIT),
    ('PhiPiPi_n', (('PhiPiPi_n', 1.0),)),
    ('PhiPiPi_c', (('PhiPiPi_c', 1.0),)),
    ('EtaGamma', (('EtaGamma', 1.0),)),
    ('EtaOmega', (('EtaOmega', 1.0),)),
    ('OmPiPi_n', (('OmPiPi_n', 1.0),)),
    ('OmPiPi_c', (('OmPiPi_c', 1.0),)),
    ('Pi0_gamma', (('PiGamma', 1.0),)),
    ('Pip_Pim_Pi0', (('3pi', 1.0),)),
    ('ppbar', (('ppbar', 1.0),)),
    ('nnbar', (('nnbar', 1.0),)),
)
KAPPA_ROWS = {
    'three_pion': ('Pip_Pim_Pi0',),
    'kkpi': ('KKpi_0_KL_KS_Pi0', 'Kp_Km_Pi0', *(f'KKpi_2_{s}' for s, _ in _CONJUGATE_SPLIT)),
}

# rho line shape of the 3pi row; EQUAL_PION_MASS is the common pion mass of the Bose-symmetry check.
RHO_MASS, RHO_WIDTH, EQUAL_PION_MASS = 0.775, 0.147, 0.139
# Signs of the rho propagators recoiling against (pi+, pi-, pi0), i.e. of BW(s_-0), BW(s_+0), BW(s_+-).
THREE_PION_SIGNS = (1, 1, 1)


# ---------------------------------------------------------------- SciPy quadrature port
def _quadrature(func, a, b, args=(), tol=1.49e-8, rtol=1.49e-8, maxiter=50, vec_func=True, miniter=1):
    """Port of scipy.integrate.quadrature (removed in SciPy 1.15): Gauss-Legendre of rising order
    until two successive estimates differ by less than tol or rtol*|value|. vec_func=True only."""
    import numpy as np
    from scipy.special import roots_legendre

    if not vec_func:
        raise NotImplementedError('only vectorized integrands are used by DeLiVeR')

    if not isinstance(args, tuple):
        args = (args,)
    value, error = np.inf, np.inf
    maxiter = max(miniter + 1, maxiter)

    for order in range(miniter, maxiter + 1):
        x, w = roots_legendre(order)
        y = (b - a) * (np.real(x) + 1) / 2.0 + a
        new = (b - a) / 2.0 * np.sum(w * np.asarray(func(y, *args)), axis=-1)
        error, value = abs(new - value), new
        if error < tol or error < rtol * abs(value):
            break
    else:
        warnings.warn(f'quadrature: maxiter ({maxiter}) exceeded, latest difference {error:e}')

    return value, error


def install_quadrature_shim():
    import scipy.integrate
    if hasattr(scipy.integrate, 'quadrature'):
        return False
    scipy.integrate.quadrature = _quadrature
    return True


# ---------------------------------------------------------------- DeLiVeR evaluation
_MODULES = {}


def load_deliver():
    """Import the pinned DeLiVeR form factors once and reset them to the B-L charges."""

    if _MODULES:
        return _MODULES
    install_quadrature_shim()

    if str(DELIVER) not in sys.path:
        sys.path.insert(0, str(DELIVER))
    names = {spec[0] for spec in (*SUBMODES.values(), *ISOVECTOR.values())}

    for name in sorted(names):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            module = importlib.import_module('src.form_factors.' + name)
        expected = (DELIVER / 'src' / 'form_factors' / (name + '.py')).resolve()
        if Path(module.__file__).resolve() != expected:
            raise RuntimeError(f'{name} resolved to {module.__file__}, not {expected}')
        module.resetParameters(1.0, 0.0, 0.0, 0.0, *CHARGES)
        _MODULES[name] = module

    return _MODULES


def _call(spec, mass):
    module, function, mode = spec
    fn = getattr(load_deliver()[module], function)
    value = float(fn(mass) if mode is None else fn(mass, mode))
    if not math.isfinite(value):
        raise RuntimeError(f'DeLiVeR {module}.{function}({mass}, {mode}) is not finite')
    return value


def submode_widths(mass):
    """Native DeLiVeR Gamma/g_BL^2 of every isoscalar submode, and the isovector channels (expected 0)."""
    return ({name: _call(spec, mass) for name, spec in SUBMODES.items()},
            {name: _call(spec, mass) for name, spec in ISOVECTOR.items()})


def evaluate_masses(masses, processes=8, pool=None):
    masses = [float(m) for m in masses]

    if pool is not None:
        return pool.map(submode_widths, masses, chunksize=8)

    if processes <= 1:
        return [submode_widths(m) for m in masses]

    with get_context('spawn').Pool(processes, initializer=load_deliver) as pool:
        result = pool.map(submode_widths, masses, chunksize=8)
        pool.close()
        pool.join()

    return result


# ---------------------------------------------------------------- table content
def generator_masses():
    sys.path.insert(0, str(ROOT))
    from exhad.core.kinematics import PARTICLES
    return PARTICLES


def dark_photon_rows():
    return {row[0]: row for row in json.loads(DP_TABLE.read_text())}


def row_thresholds(pdgs_by_label):
    particles = generator_masses()
    # Same expression as exHad's threshold rule: the plain sum of the row's generator masses.
    return {label: sum(particles[p][0] for p in pdgs if p != -999) for label, pdgs in pdgs_by_label.items()}


def row_value(label, splits, thresholds, mass, widths):
    if mass <= thresholds[label]:
        return 0.0
    return math.fsum(fraction * max(widths[sub], 0.0) for sub, fraction in splits)


def mass_grid(thresholds, extra=()):
    nodes = {round(k * 1e-3, 12) for k in range(round(TABLE_START * 1e3), round(TABLE_END * 1e3) + 1)}
    nodes = sorted(nodes)
    exact = sorted(set(t for t in (*thresholds.values(), *extra) if TABLE_START < t < TABLE_END))
    # a threshold (or the matching mass) replaces any regular node closer than 1e-7 GeV, so each is itself a node
    kept = [m for m in nodes if all(abs(m - t) >= 1e-7 for t in exact)]
    return sorted(kept + exact)


def three_pion_expression(m_plus, m_minus, m_zero, signs=THREE_PION_SIGNS):
    """|M|^2(mLLP, E1, E3) of V -> pi+ pi- pi0 in the dark-photon row family, at the given pion masses.

    4 m^2 |p+ x p-|^2 |c1/D(E1) + c2/D(E2) + c3/D(E3)|^2, D(E_k) = M^2 - s_k - i M Gamma, s_k = m^2 + m_k^2 - 2 m E_k
    the invariant mass squared of the pair recoiling against particle k (1 = pi+, 2 = pi-, 3 = pi0),
    E2 = m - E1 - E3, (c1, c2, c3) = signs. With the generator's masses the Gram factor |p+ x p-|^2 is
    non-negative on the whole Dalitz region.
    """
    import sympy as sp
    m, e1, e3 = sp.symbols('mLLP E1 E3', real=True)
    e2 = m - e1 - e3
    q1, q2, q3 = e1**2 - m_plus**2, e2**2 - m_minus**2, e3**2 - m_zero**2
    gram = sp.expand(q1 * q2 - ((q3 - q1 - q2) / 2)**2)

    def pieces(mk, ek):
        return sp.expand(RHO_MASS**2 - (m**2 + mk**2 - 2 * m * ek)), RHO_MASS * RHO_WIDTH

    (r1, i1), (r2, i2), (r3, i3) = pieces(m_plus, e1), pieces(m_minus, e2), pieces(m_zero, e3)
    d1, d2, d3 = r1 - sp.I * i1, r2 - sp.I * i2, r3 - sp.I * i3
    c1, c2, c3 = (int(c) for c in signs)
    numerator = sp.expand(c1 * d2 * d3 + c2 * d1 * d3 + c3 * d1 * d2)
    re, im = sp.re(numerator), sp.im(numerator)
    expression = (4 * m**2 * gram * sp.expand(re**2 + im**2)
                  / (sp.expand(r1**2 + i1**2) * sp.expand(r2**2 + i2**2) * sp.expand(r3**2 + i3**2)))
    return str(expression)


def _dalitz_energies(rng, mass, m1, m2, m3, n):
    """Uniform Dalitz points (E1, E3) inside the physical region of masses (m1, m2, m3)."""
    s12 = rng.uniform((m1 + m2)**2, (mass - m3)**2, 6 * n)
    s23 = rng.uniform((m2 + m3)**2, (mass - m1)**2, 6 * n)
    e1 = (mass**2 + m1**2 - s23) / (2 * mass)
    e3 = (mass**2 + m3**2 - s12) / (2 * mass)
    e2 = mass - e1 - e3
    q1, q2, q3 = e1**2 - m1**2, e2**2 - m2**2, e3**2 - m3**2
    inside = (q1 > 0) & (q2 > 0) & (q3 > 0) & (q1 * q2 - ((q3 - q1 - q2) / 2)**2 > 0)

    return e1[inside][:n], e3[inside][:n]


def verify_three_pion_family(dp_expression, expression):
    """(a) The dark-photon table's 3pi row is the table expression. (b) The expression is Bose symmetric: at
    equal pion masses |M|^2 is invariant under every permutation of the three pion energies."""
    import itertools
    import numpy as np
    import sympy as sp

    if dp_expression != expression:
        raise RuntimeError('the dark-photon Pip_Pim_Pi0 row differs from the three-pion table expression')
    m, e1, e3 = sp.symbols('mLLP E1 E3')
    equal = (EQUAL_PION_MASS,) * 3
    family = sp.lambdify((m, e1, e3), sp.sympify(three_pion_expression(*equal), locals={'mLLP': m, 'E1': e1, 'E3': e3}), 'numpy')
    rng = np.random.default_rng(11)
    symmetric = 0.0

    for mass in (0.45, 0.6, 0.782, 1.0, 1.019, 1.4, 1.9):
        a, b = _dalitz_energies(rng, mass, *equal, 400)
        energies = (a, mass - a - b, b)  # (E1, E2, E3)
        base = family(mass, a, b)
        scale = float(np.max(np.abs(base)))
        for perm in itertools.permutations(range(3)):
            symmetric = max(symmetric, float(np.max(np.abs(family(mass, energies[perm[0]], energies[perm[2]]) - base)) / scale))

    if symmetric > 1e-9:
        raise RuntimeError(f'three-pion |M|^2 is not Bose symmetric at equal pion masses ({symmetric:.2e})')

    return {'dark_photon_row_equals_table_expression': True,
            'table_sign_pattern_permutation_symmetry_max_relative_difference_at_equal_masses': symmetric}


# ---------------------------------------------------------------- matching mass
def matching_mass(pdgs_by_label, thresholds):
    """Last crossing of sum(rows) with exHad's perturbative B-L width, from direct DeLiVeR evaluation."""
    from scipy.optimize import brentq
    sys.path.insert(0, str(ROOT))
    from exhad.b_l import rates

    def excess(mass):
        widths, _ = submode_widths(mass)
        rows = math.fsum(row_value(label, splits, thresholds, mass, widths) for label, splits in ROWS)
        return rows - rates.total_width(mass)

    lo, hi = MATCH_BRACKET
    if not excess(lo) > 0 > excess(hi):
        raise RuntimeError('sum of exclusive rows does not cross the perturbative width in the bracket')
    return brentq(excess, lo, hi, xtol=1e-10, rtol=1e-14)


# ---------------------------------------------------------------- output
def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def deliver_revision():
    env = {**os.environ, 'GIT_OPTIONAL_LOCKS': '0'}  # read-only: no index refresh in the pinned checkout
    head = subprocess.check_output(['git', '-C', str(DELIVER), 'rev-parse', 'HEAD'], text=True, env=env).strip()
    dirty = subprocess.check_output(['git', '-C', str(DELIVER), 'diff', '--name-only', 'HEAD', '--', 'src'],
                                    text=True, env=env)
    if head != DELIVER_REVISION or dirty.strip():
        raise RuntimeError(f'{DELIVER} differs from the pinned DeLiVeR revision {DELIVER_REVISION}')
    return head


def build(processes=8):
    revision = deliver_revision()
    dp = dark_photon_rows()
    missing = [label for label, _ in ROWS if label not in dp]
    if missing:
        raise RuntimeError(f'labels absent from the dark-photon table: {missing}')
    pdgs = {label: [int(p) for p in dp[label][1]] for label, _ in ROWS}
    thresholds = row_thresholds(pdgs)
    particles = generator_masses()
    three_pion = three_pion_expression(particles[211][0], particles[-211][0], particles[111][0], THREE_PION_SIGNS)
    family_check = verify_three_pion_family(dp['Pip_Pim_Pi0'][3], three_pion)

    def row_values(mass, widths):
        return [row_value(label, splits, thresholds, mass, widths) for label, splits in ROWS]

    # the matching mass is a grid node, so the interpolated row sum equals exHad's B-L width there
    mx = matching_mass(pdgs, thresholds)
    base = mass_grid(thresholds, extra=(mx,))
    added = []
    with get_context('spawn').Pool(max(processes, 1), initializer=load_deliver) as pool:
        native = dict(zip(base, evaluate_masses(base, pool=pool)))
        pending = list(zip(base[:-1], base[1:]))
        for _ in range(REFINE_PASSES):
            mids = [round(0.5 * (a + b), 15) for a, b in pending]
            if not mids:
                break
            native.update(zip(mids, evaluate_masses(mids, pool=pool)))
            next_pending = []
            for (a, b), c in zip(pending, mids):
                va, vb, vc = (row_values(x, native[x][0]) for x in (a, b, c))
                scale = REFINE_TOL * (math.fsum(vc) + c / (12 * math.pi))
                linear = [0.5 * (x + y) for x, y in zip(va, vb)]
                miss = max(max(abs(l - v) for l, v in zip(linear, vc)), abs(math.fsum(linear) - math.fsum(vc)))
                if miss > scale:
                    added.append(c)
                    next_pending += [(a, c), (c, b)]
            pending = next_pending
        pool.close()
        pool.join()
    unresolved = pending
    grid = sorted(base + added)
    evaluated = [native[m] for m in grid]
    isovector_max = max(abs(v) for _, iso in evaluated for v in iso.values())
    if isovector_max != 0.0:
        raise RuntimeError(f'an isovector DeLiVeR channel couples to the B-L current ({isovector_max})')
    negative = [(m, sub, w) for m, (widths, _) in zip(grid, evaluated) for sub, w in widths.items() if w < 0]
    removed = {}  # DeLiVeR width at or below the generator threshold, set to zero by the threshold rule
    rows = []
    for label, splits in ROWS:
        points = []
        for mass, (widths, _) in zip(grid, evaluated):
            value = row_value(label, splits, thresholds, mass, widths)
            if mass <= thresholds[label]:
                spill = math.fsum(f * widths[s] for s, f in splits)
                if spill > 0:
                    removed[label] = max(removed.get(label, 0.0), spill)
            points.append([mass, value])
        expression = three_pion if label == 'Pip_Pim_Pi0' else dp[label][3]
        rows.append([label, pdgs[label], points, expression])
    total = [math.fsum(row[2][i][1] for row in rows) for i in range(len(grid))]

    rates_inputs = json.loads((OUT / 'rates.json').read_text())

    OUT.mkdir(parents=True, exist_ok=True)
    TABLE.write_text(json.dumps(rows, separators=(',', ':')))
    WIDTHS.write_text(''.join(f'{m!r}\t{w!r}\n' for m, w in zip(grid, total)))
    import numpy
    import scipy
    import sympy
    meta = {
        'schema': 'exhad-bl-low-mass-v1',
        'purpose': 'B-L exclusive hadronic rows below 2 GeV in EventCalc row format',
        'units': 'GeV; row node values and hadronic widths are partial widths per g_BL^2 (not branching ratios)',
        'lagrangian': 'L = g_BL Z\'_mu J_{B-L}^mu, quark charges q_u = q_d = q_s = 1/3; '
                      'Gamma(Z\' -> l+ l-)/g^2 = m/(12 pi)(1+2x)sqrt(1-4x), Gamma(Z\' -> nu nubar)/g^2 = m/(24 pi) per flavour',
        'table': TABLE.name,
        'hadronic_widths': {'file': WIDTHS.name, 'columns': ['mass_gev', 'sum_of_rows_width_per_g2_gev']},
        'interpolation': 'piecewise-linear-no-extrapolation; thresholds are grid nodes',
        'threshold_rule': 'each row is zero at and below the sum of exhad.core.kinematics.PARTICLES masses',
        'grid': {'start_gev': grid[0], 'end_gev': grid[-1], 'nodes': len(grid), 'base_step_gev': BASE_STEP_MEV * 1e-3,
                 'refinement': {'tolerance': REFINE_TOL, 'criterion': 'midpoint |linear - native| of every row '
                                'and of the row sum <= tolerance x (sum of rows + m/(12 pi))',
                                'passes': REFINE_PASSES, 'nodes_added': len(added),
                                'intervals_unresolved_at_smallest_step': [list(i) for i in unresolved]},
                 'threshold_nodes_gev': sorted(set(thresholds.values()))},
        'rows': {label: {'deliver_submodes': {s: f for s, f in splits}, 'threshold_gev': thresholds[label]}
                 for label, splits in ROWS},
        'isovector_channels_zero': sorted(ISOVECTOR),
        'matching_mass_gev': mx,
        'matching_mass_is_grid_node': mx in grid,
        'matching_definition': 'last crossing on [1.70, 1.80] GeV of the sum of rows (direct DeLiVeR) with '
                               'exhad.b_l.rates.total_width; join the hadronic width there with a C2 join',
        'kappa': {
            'three_pion': {'value': rates_inputs['three_pion_scale'], 'rows': list(KAPPA_ROWS['three_pion']),
                           'source': 'data/b-l/rates.json three_pion_scale'},
            'kkpi': {'value': rates_inputs['kkpi_scale'], 'rows': list(KAPPA_ROWS['kkpi']),
                     'source': 'data/b-l/rates.json kkpi_scale'},
            'law': 'on [m_x, 2.0) GeV scale the kappa rows by 1 + S(m)(kappa - 1), S the quintic smoothstep '
                   'from 0 at m_x to 1 at 2.0 GeV; no scaling below m_x',
        },
        'three_pion_matrix_element': {
            'row': 'Pip_Pim_Pi0',
            'form': '4 m^2 |p+ x p-|^2 |1/D(E1) + 1/D(E2) + 1/D(E3)|^2, D(E_k) = M^2 - (m^2 + m_k^2 - 2 m E_k) - i M Gamma, '
                    'E2 = m - E1 - E3; particles 1, 2, 3 = pi+, pi-, pi0',
            'sign_convention': 'the rho propagators BW(s_-0), BW(s_+0), BW(s_+-) enter with the same sign (Bose '
                               'symmetry of the I = 0 three-pion state; hep-ph/0512180, the model of the DeLiVeR 3pi '
                               'rates); the dark-photon table row carries the same expression.',
            'rho_signs_recoiling_against_pip_pim_pi0': list(THREE_PION_SIGNS),
            'rho_mass_gev': RHO_MASS, 'rho_width_gev': RHO_WIDTH,
            'pion_masses_gev': {'211': particles[211][0], '-211': particles[-211][0], '111': particles[111][0]},
            'checks': family_check,
        },
        'clipped_negative_deliver_widths': len(negative),
        'deliver_width_removed_at_or_below_threshold_max_per_g2_gev': removed,
        'sources': {
            'deliver_revision': revision,
            'deliver_form_factor_sha256': {name: sha256(DELIVER / 'src' / 'form_factors' / (name + '.py'))
                                           for name in sorted({s[0] for s in SUBMODES.values()})},
            'dark_photon_table_sha256': sha256(DP_TABLE),
            'rates_json_sha256': sha256(OUT / 'rates.json'),
            'kinematics_sha256': sha256(ROOT / 'exhad' / 'core' / 'kinematics.py'),
            'builder_sha256': sha256(Path(__file__)),
            'quadrature': 'tools/build_bl_low_mass.py _quadrature (port of the removed scipy.integrate.quadrature)',
            'python': platform.python_version(), 'numpy': numpy.__version__, 'scipy': scipy.__version__,
            'sympy': sympy.__version__,
        },
        'outputs_sha256': {TABLE.name: sha256(TABLE), WIDTHS.name: sha256(WIDTHS)},
    }
    META.write_text(json.dumps(meta, indent=1) + '\n')
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--processes', type=int, default=8)
    args = parser.parse_args()
    meta = build(args.processes)
    print(f"wrote {TABLE.relative_to(ROOT)}: {len(ROWS)} rows, {meta['grid']['nodes']} nodes "
          f"{meta['grid']['start_gev']}-{meta['grid']['end_gev']} GeV; m_x = {meta['matching_mass_gev']:.6f} GeV")


if __name__ == '__main__':
    main()
