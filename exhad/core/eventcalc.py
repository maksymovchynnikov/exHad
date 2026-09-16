"""EventCalc terminal convention for complete rest-frame events.

pi0 -> gamma gamma; K_S -> pi+ pi- with probability 0.692, otherwise pi0 pi0 -> 4 gamma.
Each decay is isotropic in the parent rest frame and boosted by the parent velocity.
The decay generator is separate from the hadronizer stream: decay_rng(seed), or a
seeded random.Random for B-L and alp-fermion open charm, whose streams use correctly rounded
sums (fsum=True: math.fsum, math.hypot, math.cos/sin).
"""
import math
import numpy as np
from .event_model import MASS_SHELL_ABS_TOL_GEV2
from .pdg import antiparticle, m0

_MPI, _MPI0 = m0(211), m0(111)


def decay_rng(seed):
    return np.random.default_rng(np.random.SeedSequence([int(seed), 0x45434445]))


def _two_body(parent, m1, m2, id1, id2, rng, fsum):
    """Isotropic decay of parent [px, py, pz, E, m] into two six-field particles."""
    px, py, pz, e, m = parent
    pstar = np.sqrt(max((m * m - (m1 + m2) ** 2) * (m * m - (m1 - m2) ** 2), 0.0)) / (2 * m)
    c = rng.uniform(-1, 1)
    s = np.sqrt(1 - c * c)
    ph = rng.uniform(0, 2 * np.pi)
    cos, sin = (math.cos, math.sin) if fsum else (np.cos, np.sin)
    p1 = np.array([pstar * s * cos(ph), pstar * s * sin(ph), pstar * c])
    beta = np.array([px, py, pz]) / e
    b2 = math.fsum(beta * beta) if fsum else float(beta @ beta)
    out = []

    for pv, mm, pid in ((p1, m1, id1), (-p1, m2, id2)):
        ee = math.hypot(pstar, mm) if fsum else np.sqrt(pstar ** 2 + mm * mm)
        if b2 > 1e-16:
            gamma = 1.0 / np.sqrt(1 - b2)
            bp = math.fsum(beta * pv) if fsum else float(beta @ pv)
            pv = pv + beta * (gamma - 1.0) * bp / b2 + gamma * beta * ee
            ee = gamma * (ee + bp)
        out.append([pv[0], pv[1], pv[2], ee, mm, float(pid)])

    return out


def decay(particle, rng, fsum=False):
    """Six-field particle [px, py, pz, E, m, pdg] after the EventCalc pi0 and K_S decays."""
    pid = int(particle[5])

    if pid == 111:
        return _two_body(particle[:5], 0.0, 0.0, 22, 22, rng, fsum)

    if pid == 310:
        if rng.random() < 0.692:
            return _two_body(particle[:5], _MPI, _MPI, 211, -211, rng, fsum)
        return [q for p in _two_body(particle[:5], _MPI0, _MPI0, 111, 111, rng, fsum) for q in decay(p, rng, fsum)]

    return [particle]


def _mass_squared(momentum):
    return (momentum.energy_gev * momentum.energy_gev - momentum.px_gev * momentum.px_gev
            - momentum.py_gev * momentum.py_gev - momentum.pz_gev * momentum.pz_gev)


def spacelike(event):
    """Whether a stable particle lies below the event-record mass-shell tolerance."""
    return any(_mass_squared(p.momentum) < -MASS_SHELL_ABS_TOL_GEV2 for p in event.stable_particles)


def six_field(event, rng, conjugate=False, fsum=False):
    """Flat EventCalc record of a complete event: its stable particles with pi0 and K_S decayed.

    conjugate replaces every particle by its antiparticle before the decays.
    """
    if spacelike(event):
        raise RuntimeError('matched event contains a spacelike terminal particle')
    rows = []
    for particle in event.stable_particles:
        p, pdg = particle.momentum, int(particle.pdg_id)
        pdg = antiparticle(pdg) if conjugate else pdg
        mass = float(np.sqrt(max(0.0, _mass_squared(p))))
        rows += decay([p.px_gev, p.py_gev, p.pz_gev, p.energy_gev, mass, float(pdg)], rng, fsum)
    return [value for row in rows for value in row]
