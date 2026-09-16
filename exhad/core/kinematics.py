"""Rest-frame primary decay kinematics derived from EventCalc (BSD-3-Clause, LICENSE-EventCalc).

Particles are 8-field rows [px, py, pz, E, m, pdg, charge, stability]; exHad writes stability 0 and
never reads it. Angles come
from numba's generator; Dalitz energies and their accept-reject uniforms from numpy's
global generator. seed() seeds both once per table channel.
"""
from functools import lru_cache
import math
import re
import numba as nb
import numpy as np

# Fictitious lightest-meson masses of partons; a parton pair must reach their sum.
FICTITIOUS = {1: 0.1396, 2: 0.1396, 21: 0.1396, 3: 0.496, 4: 1.875, 5: 5.280}
FLAT = frozenset({None, "", "-", "1.", "1.0", 1.})  # matrix-element entries of flat phase space (the scalar table stores 1.0)


@nb.njit
def _seed_numba(seed):
    np.random.seed(seed)


def seed(value):
    np.random.seed(int(value))
    _seed_numba(int(value))


def _two_body(m, size, m1, m2, pdg1, pdg2, charge1, charge2, stability1, stability2):
    """size isotropic two-body decays at rest, 16 fields per event."""
    products = np.empty((size, 16), dtype=np.float64)
    E1 = (m**2 + m1**2 - m2**2) / (2 * m)
    E2 = (m**2 + m2**2 - m1**2) / (2 * m)
    pmod = np.sqrt(E1**2 - m1**2)

    for i in range(size):
        theta = np.arccos(np.random.uniform(-1, 1))
        phi = np.random.rand() * 2 * np.pi
        px = pmod * np.sin(theta) * np.cos(phi)
        py = pmod * np.sin(theta) * np.sin(phi)
        pz = pmod * np.cos(theta)
        products[i] = np.array([px, py, pz, E1, m1, pdg1, charge1, stability1,
                                -px, -py, -pz, E2, m2, pdg2, charge2, stability2])

    return products


def __getattr__(name):
    """Compile two_body at first use (its fixed signature casts the arguments); importing never pays it."""
    if name != 'two_body':
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    compiled = globals()['two_body'] = nb.njit(
        '(float64, int64, float64, float64, int64, int64, int64, int64, int64, int64,)')(_two_body)
    return compiled


def dalitz(m, m1, m2, m3, n, pdgs=(0, 0, 0)):
    """n flat Dalitz points (E1, E3) inside the physical region.

    With two partons the non-parton (first or last) energy is bounded so that the pair
    reaches its fictitious-meson threshold; that branch keeps a fixed batch size.
    """
    hi1 = (m**2 + m1**2 - (m2 + m3)**2) / (2 * m)
    hi3 = (m**2 + m3**2 - (m1 + m2)**2) / (2 * m)
    partons = [abs(p) in FICTITIOUS for p in pdgs]
    adaptive, last = sum(partons) < 2, False
    if not adaptive:
        if partons[0] and partons[2]:
            raise ValueError('Unexpected scenario: non-parton is pdg2, which is not anticipated.')
        last = partons[0]
        (a, b), mass, limit = (pdgs[:2], m3, hi3) if last else (pdgs[1:], m1, hi1)
        bound = min((m**2 + mass**2 - (FICTITIOUS[abs(a)] + FICTITIOUS[abs(b)])**2) / (2 * m), limit)
        hi1, hi3 = (hi1, bound) if last else (bound, hi3)
    e1, e3 = np.zeros(n), np.zeros(n)
    found, rate = 0, 1.0
    while found < n:
        size = int(1.2 * (n - found) / min(max(rate, 0.001), 1.0))
        if last:
            E3 = np.random.uniform(m3, hi3, size)
            E1 = np.random.uniform(m1, hi1, size)
        else:
            E1 = np.random.uniform(m1, hi1, size)
            E3 = np.random.uniform(m3, hi3, size)
        E2 = m - E1 - E3
        valid = np.logical_and(E2 > m2, (E2**2 - m2**2 - (E1**2 - m1**2) - (E3**2 - m3**2))**2
                               < 4 * (E1**2 - m1**2) * (E3**2 - m3**2))
        count = np.sum(valid)
        k = min(count, n - found)
        e1[found:found + k] = E1[valid][:k]
        e3[found:found + k] = E3[valid][:k]
        if adaptive:
            rate = count / len(valid)
        found += k
    return e1, e3


@nb.njit
def orient(energies, m, m1, m2, m3, pdg1, pdg2, pdg3, charge1, charge2, charge3,
           stability1, stability2, stability3):
    """One three-body event at Dalitz point (E1, E3): p1 along a Haar-random axis, rolled by kappa."""
    E1, E3 = energies[0], energies[1]
    theta = np.arccos(np.random.uniform(-1, 1))
    phi = np.random.uniform(-np.pi, np.pi)
    kappa = np.random.uniform(-np.pi, np.pi)
    # Expressions generated from EventCalc's rotation matrices, in EventCalc's operation order.
    B = E1**2 - m1**2
    A2 = -E1**2 + E3**2 + m1**2 + m2**2 - m3**2 - (-E1 - E3 + m)**2
    C2 = -m2**2 + (-E1 - E3 + m)**2
    s2 = np.sqrt(1 - 1 / 4 * A2**2 / (B * C2))
    A3 = -E1**2 - E3**2 + m1**2 - m2**2 + m3**2 + (-E1 - E3 + m)**2
    C3 = E3**2 - m3**2
    s3 = np.sqrt(1 - 1 / 4 * A3**2 / (B * C3))
    sin_t, cos_t, sin_p, cos_p = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
    sin_k, cos_k = np.sin(kappa), np.cos(kappa)

    return np.array([
        np.sqrt(B) * sin_p * sin_t, np.sqrt(B) * sin_t * cos_p, np.sqrt(B) * cos_t, E1, m1, pdg1, charge1, stability1,
        s2 * np.sqrt(C2) * sin_k * cos_p + s2 * np.sqrt(C2) * sin_p * cos_k * cos_t + (1 / 2) * A2 * sin_p * sin_t / np.sqrt(B),
        -s2 * np.sqrt(C2) * sin_k * sin_p + s2 * np.sqrt(C2) * cos_k * cos_p * cos_t + (1 / 2) * A2 * sin_t * cos_p / np.sqrt(B),
        -s2 * np.sqrt(C2) * sin_t * cos_k + (1 / 2) * A2 * cos_t / np.sqrt(B), m - E1 - E3, m2, pdg2, charge2, stability2,
        -s3 * np.sqrt(C3) * sin_k * cos_p - s3 * np.sqrt(C3) * sin_p * cos_k * cos_t + (1 / 2) * A3 * sin_p * sin_t / np.sqrt(B),
        s3 * np.sqrt(C3) * sin_k * sin_p - s3 * np.sqrt(C3) * cos_k * cos_p * cos_t + (1 / 2) * A3 * sin_t * cos_p / np.sqrt(B),
        s3 * np.sqrt(C3) * sin_t * cos_k + (1 / 2) * A3 * cos_t / np.sqrt(B), E3, m3, pdg3, charge3, stability3])


@lru_cache(maxsize=256)
def envelope(expression, m, pdgs, masses):
    """Deterministic accept-reject envelope of |M|^2 on dalitz's region: (|M|^2, envelope, acceptance).

    The region (with dalitz's parton-pair threshold cut) is mapped onto E3 and the fraction t of the
    analytic E1 interval at that E3, so thin near-threshold slivers stay resolved. f = |M|^2 is
    evaluated on 256x256 (E3, t) cell centres; the eight largest are polished by ten 3x zoom rounds
    that reach the boundary, and the envelope is 1.05 times the largest value found. Negative f is
    clipped to 0 as a guard, however large its part of the region: max(f, 0) is sampled. Closed
    regions, non-finite f and f <= 0 at every cell centre raise. acceptance is the expected accepted fraction of flat dalitz points.
    """
    m1, m2, m3 = masses
    top1, top3 = math.inf, (m**2 + m3**2 - (m1 + m2)**2) / (2 * m)
    partons = [abs(p) in FICTITIOUS for p in pdgs]
    if sum(partons) >= 2 and not (partons[0] and partons[2]):
        (a, b), mass = (pdgs[:2], m3) if partons[0] else (pdgs[1:], m1)
        bound = (m**2 + mass**2 - (FICTITIOUS[abs(a)] + FICTITIOUS[abs(b)])**2) / (2 * m)
        top1, top3 = (top1, min(bound, top3)) if partons[0] else (bound, top3)
    function = matrix_element(expression, masses)

    def values(e3, t):
        """f at (E1, E3) = (low + t*width, E3) and the width of the allowed E1 interval at E3."""
        e3, t = (np.ravel(x) for x in np.meshgrid(np.clip(e3, m3, top3), np.clip(t, 0., 1.)))

        with np.errstate(divide='ignore', invalid='ignore'):
            s = m**2 + m3**2 - 2 * m * e3
            star = (s + m1**2 - m2**2) / (2 * np.sqrt(s))
            spread = np.sqrt(np.maximum(e3**2 - m3**2, 0.) * np.maximum(star**2 - m1**2, 0.))
            low = ((m - e3) * star - spread) / np.sqrt(s)
            width = np.minimum(((m - e3) * star + spread) / np.sqrt(s), top1) - low
            open_ = width > 0
            e1, e3, t, width = low[open_] + t[open_] * width[open_], e3[open_], t[open_], width[open_]

        return e3, t, width, np.broadcast_to(np.asarray(function(m, e1, e3), dtype=float), e1.shape)

    cell = (np.arange(256) + .5) / 256
    e3, t, width, f = values(m3 + (top3 - m3) * cell, cell) if top3 > m3 else ((),) * 4
    if not len(f):
        raise ValueError(f'Three-body region of {list(pdgs)} is closed at {m:g} GeV')
    if not np.all(np.isfinite(f)):
        raise ValueError(f'Non-finite three-body |M|^2 for {list(pdgs)} at {m:g} GeV')
    peak = f.max()
    if peak <= 0:
        raise ValueError(f'Three-body |M|^2 of {list(pdgs)} at {m:g} GeV is not positive anywhere in its region')
    grid = np.linspace(-1., 1., 9)
    for i in np.argsort(f)[-8:]:
        x3, xt, step3, stept = e3[i], t[i], (top3 - m3) / 256, 1 / 256
        for _ in range(10):
            p3, pt, _, v = values(x3 + step3 * grid, xt + stept * grid)
            finite = np.flatnonzero(np.isfinite(v))  # an exact massless endpoint is 0/0
            if finite.size and v[finite].max() >= peak:
                j = finite[v[finite].argmax()]
                peak, x3, xt = v[j], p3[j], pt[j]
            step3, stept = step3 / 3, stept / 3
    return function, 1.05 * peak, (np.maximum(f, 0.) @ width) / width.sum() / (1.05 * peak)


def three_body(m, count, pdgs, masses, charges, stabilities, expression):
    """count three-body events distributed as phase space times max(|M|^2, 0), by exact accept-reject.

    Flat dalitz points are accepted with probability max(f, 0)/envelope, so the law is independent of
    count. A point above the envelope, or 50x the expected proposals, raises instead of biasing.
    """
    function, bound, acceptance = envelope(expression, float(m), tuple(pdgs), tuple(masses))
    points, proposals = np.empty((0, 2)), 0
    while len(points) < count:
        size = min(int(1.2 * (count - len(points)) / acceptance) + 1, 1 << 20)
        e1, e3 = dalitz(m, *masses, size, pdgs)
        f = np.broadcast_to(np.asarray(function(m, e1, e3), dtype=float), e1.shape)
        if not np.all(f <= bound):  # also catches NaN
            raise RuntimeError(f'Three-body |M|^2 of {list(pdgs)} at {m:g} GeV is non-finite or exceeds its envelope')
        keep = np.flatnonzero(np.random.uniform(0., bound, size) < f)[:count - len(points)]
        points = np.concatenate([points, np.column_stack([e1[keep], e3[keep]])])
        proposals += size
        if proposals > 50 * count / acceptance + 1e5 and len(points) < count:
            raise RuntimeError(f'Three-body accept-reject for {list(pdgs)} at {m:g} GeV exceeded its proposal cap')
    return np.array([orient(e, m, *masses, *pdgs, *charges, *stabilities) for e in points])


def _breakup(a, b, c, kinetic):
    """Two-body breakup momentum of mass a into b + c, with kinetic = a - b - c passed exactly."""
    return np.sqrt(kinetic * (a + b + c) * (kinetic + 2 * c) * (kinetic + 2 * b)) / (2 * a)


@lru_cache(maxsize=256)
def phase_space_bound(m, masses):
    """Proven maximum of the Raubold-Lynch weight W = prod_k p*(M_k; M_{k-1}, m_k) of flat n-body phase space.

    The intermediate masses are M_k = sum_{j<=k} m_j + t_1 + ... + t_k with kinetic splits t >= 0
    summing to T = m - sum(masses). log p*(a; b, c) = sum log(a +- b +- c)/2 - log a is jointly concave
    in (a, b) for a > b + c: the -log a curvature 1/a^2 is dominated because 1/A + 1/B <= a^2 + b^2 + c^2
    <= 2a^2, A and B being the curvatures of the (a - b) and (a + b) factor pairs. So log W is concave
    in t and, at any interior t0 with gradient g, log W <= log W(t0) + max_k T g_k - g.t0. Exponentiated
    gradient ascent shrinks that gap. Returns min(that certificate, GENBOD's bound prod_k p*(m_k + T +
    sum_{j<k} m_j; sum_{j<k} m_j, m_k)), times 1 + 1e-9 for rounding.
    """
    mu, n = np.asarray(masses, dtype=float), len(masses)
    cum = np.cumsum(mu)
    T = m - cum[-1]
    genbod = math.prod(_breakup(cum[k] + T, cum[k - 1], mu[k], T) for k in range(1, n))

    def ascent_state(u):
        """log W, its gradient in the fractions u = t/T, and the certified gap at u."""
        t = T * u
        M = np.concatenate([[mu[0]], cum[1:] + np.cumsum(t)])
        M[-1] = m
        a, b, c = M[1:], M[:-1], mu[1:]
        factors = (t, a + b + c, t + 2 * c, t + 2 * b)
        value = math.fsum(.5 * np.log(np.prod(factors, axis=0)) - np.log(2 * a))
        d_a = .5 * (1 / t + 1 / factors[1] + 1 / factors[2] + 1 / factors[3]) - 1 / a
        d_b = .5 * (-1 / t + 1 / factors[1] - 1 / factors[2] + 1 / factors[3])
        d_M = np.zeros(n)
        d_M[1:-1] = d_a[:-1] + d_b[1:]  # M_0 and M_{n-1} = m are fixed
        gradient = T * np.cumsum(d_M[::-1])[::-1][1:]

        return value, gradient, gradient.max() - gradient @ u

    u = np.full(n - 1, 1. / (n - 1))
    value, gradient, gap = ascent_state(u)
    step = 1. / np.abs(gradient).max() if np.abs(gradient).max() > 0 else 1.
    for _ in range(2000):
        if gap < 1e-6 or step < 1e-12:
            break
        trial = u * np.exp(step * (gradient - gradient.max()))
        trial /= trial.sum()
        state = ascent_state(trial) if np.all(trial > 0) else (-math.inf,)
        if state[0] >= value:
            u, (value, gradient, gap), step = trial, state, 1.5 * step
        else:
            step /= 2
    return min(genbod, math.exp(value + gap)) * (1 + 1e-9)


def n_body(m, count, pdgs, masses, charges, stabilities):
    """count events of exact flat Lorentz-invariant n-body phase space, as 8-field rows in the order of pdgs.

    Raubold-Lynch: sorted uniform kinetic splits give the intermediate masses, accepted with probability
    W/phase_space_bound; each accepted split is decayed sequentially with isotropic angles. Proposals come in
    fixed blocks whose random draws depend only on the block, so a batch is the prefix of any larger batch
    at the same seed. A weight above the bound raises instead of biasing.
    """
    mu, n = np.asarray(masses, dtype=float), len(pdgs)
    cum = np.cumsum(mu)
    T = m - cum[-1]
    if n < 3 or not T > 0:
        raise ValueError(f'{n}-body phase space of {list(pdgs)} is closed at {m:g} GeV')
    bound = phase_space_bound(float(m), tuple(map(float, mu)))
    blocks = [(np.empty((0, n)), np.empty((0, n - 1)), np.empty((0, n - 1, 2)))]
    while sum(len(b[0]) for b in blocks) < count:
        kinetic = T * np.sort(np.random.uniform(0., 1., (1 << 14, n - 2)), axis=1)
        split = np.column_stack([kinetic, np.full(len(kinetic), T)])
        t = np.diff(split, axis=1, prepend=0.)
        M = np.column_stack([np.full(len(t), mu[0]), cum[1:-1] + kinetic, np.full(len(t), m)])
        momenta = _breakup(M[:, 1:], M[:, :-1], mu[1:], t)
        weight = momenta.prod(axis=1)
        if not np.all(weight <= bound):  # also catches NaN
            raise RuntimeError(f'{n}-body phase-space weight of {list(pdgs)} at {m:g} GeV exceeds its bound')
        keep = np.random.uniform(0., bound, len(weight)) < weight
        blocks.append((M[keep], momenta[keep], np.random.uniform(0., 1., (keep.sum(), n - 1, 2))))
    M, momenta, angles = (np.concatenate(parts)[:count] for parts in zip(*blocks))
    events = np.empty((count, n, 8))
    events[:, :, 4:] = np.column_stack([mu, pdgs, charges, stabilities])
    frame, frame_e = np.zeros((count, 3)), np.full(count, float(m))
    for k in range(n - 1, 0, -1):
        cos = 2 * angles[:, k - 1, 0] - 1
        sin, phi = np.sqrt(1 - cos * cos), 2 * np.pi * angles[:, k - 1, 1]
        q = momenta[:, k - 1, None] * np.column_stack([sin * np.cos(phi), sin * np.sin(phi), cos])
        energy = (M[:, k]**2 + mu[k]**2 - M[:, k - 1]**2) / (2 * M[:, k])
        daughter = _boost(q, energy, frame, frame_e, M[:, k])
        frame, frame_e = _boost(-q, M[:, k] - energy, frame, frame_e, M[:, k])
        events[:, k, :3], events[:, k, 3] = daughter
    events[:, 0, :3], events[:, 0, 3] = frame, frame_e
    return events.reshape(count, 8 * n)


def _boost(p, e, frame, frame_e, frame_m):
    """Four-momenta (p, e) given in the rest frame of a system with lab momentum frame, energy frame_e, mass frame_m."""
    dot = np.einsum('ij,ij->i', frame, p)
    return p + frame * ((dot / (frame_e + frame_m) + e) / frame_m)[:, None], (frame_e * e + dot) / frame_m


@lru_cache(maxsize=256)
def matrix_element(expression, masses):
    """Compile an EventCalc three-body |M|^2(mLLP, E_1, E_3) string at the generated masses; constants give 1.

    m1, m2, m3 in the string are the masses of the row's three particles that generate the phase space.
    The real part is returned; an imaginary part above 1e-12 of it raises.
    """
    if expression in FLAT:
        return lambda _m, _e1, _e3: 1.0
    import sympy as sp
    m_llp, energy1, energy3, *particle_masses = sp.symbols("mLLP E_1 E_3 m1 m2 m3")
    rewritten = expression.replace("***", "e").replace("\\/", "/")
    rewritten = rewritten.replace("E1", "E_1").replace("E3", "E_3")
    rewritten = re.sub(r"Symbol\(\s*'([^']+)'\s*\)", r"\1", rewritten)
    rewritten = re.sub(r"Float\(\s*'([^']+)'\s*\)", r"\1", rewritten)
    rewritten = re.sub(r"Integer\(\s*(\d+)\s*\)", r"\1", rewritten)
    parsed = sp.sympify(rewritten, locals={
        "mLLP": m_llp, "E_1": energy1, "E_3": energy3, "UnitStep": sp.Heaviside})
    function = sp.lambdify((m_llp, energy1, energy3, *particle_masses), parsed, "numpy")

    def real(m, e1, e3):
        value = function(m, e1, e3, *masses)
        if not np.iscomplexobj(value):
            return value
        if np.any(np.abs(np.imag(value)) > 1e-12 * np.abs(np.real(value))):
            raise ValueError(f'Three-body |M|^2 {expression[:60]!r} has a non-negligible imaginary part')
        return np.real(value)
    return real
