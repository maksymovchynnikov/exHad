"""Realizers for exact EventCalc decay-table rows and alp-fermion boundary families.

The Model-1 sampler (sampler.py) has already drawn the outer row or conditioned family; nothing here
selects a rate. Ordinary rows use EventCalc's decay kernels (or complete
Pythia primaries for unstable products); the alp-fermion three-pion and pipi-gamma
families use a high-mass phase-space or anomalous radiative kernel.
"""

from __future__ import annotations
import atexit
import math
from pathlib import Path
from ..core.event_worker import WorkerPool
from ..core.pdg import charge, m0


class PortableModel1RealizerError(ValueError):
    """An outer row cannot be realized as a stable event."""

# Stable products realized by phase space; any other row needs complete Pythia primaries.
STABLE_WITHOUT_PYTHIA = {11, -11, 13, -13, 12, -12, 14, -14, 16, -16, 22, 211, -211, 321, -321, 130}

# Pythia runtimes leased by the Model-1 sampler and the realizers for the process lifetime.
RUNTIMES = WorkerPool(8)
atexit.register(RUNTIMES.close)


def _anomalous_pipi_gamma_weight(mass, energy1, energy3, pion_mass):
    """|p+ x p-|^2: photon-polarization sum of the pseudoscalar epsilon-tensor
    amplitude with constant reduced form factor (m^2 is constant at fixed mass).
    It vanishes in the soft and collinear limits."""
    import numpy as np
    energy2 = mass - energy1 - energy3
    pair_s = mass * mass - 2.0 * mass * energy3
    dot = energy1 * energy2 - (pair_s - 2.0 * pion_mass ** 2) / 2.0

    return np.maximum(0.0, (energy1 ** 2 - pion_mass ** 2)
                      * (energy2 ** 2 - pion_mass ** 2) - dot ** 2)


def realize_alp_fermion_boundary_conditioned_family(*, mass_gev, count, seed, row_id, configuration):
    """Realize an already selected alp-fermion three-pion or pipi-gamma family."""
    import numpy as np
    from ..core import kinematics
    from ..core.eventcalc import decay, decay_rng
    mass = float(mass_gev)
    local_seed = seed % 899_900_000 or 1
    kinematics.seed(local_seed)
    rng, terminal_rng = np.random.default_rng(local_seed), decay_rng(seed)
    modes = configuration["boundary_specification"]["families"][row_id]["modes"]
    output = []

    for _ in range(count):
        mode = modes[int(rng.choice(len(modes), p=[x["probability"] for x in modes]))]
        pdgs = tuple(mode["pdg_ids"])
        masses, charges = tuple(m0(code) for code in pdgs), tuple(charge(code) for code in pdgs)
        if math.fsum(masses) >= mass:
            raise PortableModel1RealizerError("alp-fermion conditioned mode is kinematically closed")
        if row_id == "three-pion":
            energy1, energy3 = kinematics.dalitz(mass, *masses, 1)
            energies = np.array([energy1[0], energy3[0]])
        else:
            # Exact rejection, also for one event: |p_i| <= m/2 bounds
            # |p+ x p-|^2 by m^4/16 on the whole Dalitz domain.
            energies = None
            bound = mass ** 4 / 16.0
            for _attempt in range(1024):
                energy1, energy3 = kinematics.dalitz(mass, *masses, 64)
                weights = _anomalous_pipi_gamma_weight(mass, energy1, energy3, masses[0])
                if np.any(weights > bound * (1.0 + 1.0e-12)):
                    raise PortableModel1RealizerError("radiative kernel exceeded its envelope")
                accepted = np.flatnonzero(rng.random(len(weights)) * bound < weights)
                if len(accepted):
                    index = int(accepted[0])
                    energies = np.array([energy1[index], energy3[index]])
                    break
            if energies is None:
                raise PortableModel1RealizerError("radiative phase-space rejection exhausted")
        primary = kinematics.orient(energies, mass, *masses, *pdgs, *charges, 0., 0., 0.)
        event = []
        for offset in range(0, len(primary), 8):
            particle = list(float(value) for value in primary[offset:offset + 6])
            for terminal in decay(particle, terminal_rng):
                event.extend(float(value) for value in terminal)
        output.append(event)

    return output


def realize_eventcalc_decay_table_row(*, mass_gev, count, seed, row_id, configuration):
    """Realize exactly ``row_id`` of an EventCalc decay table (all deployed rows are two-body).

    Phase space when both products are stable; otherwise complete Pythia
    primaries, with (c, cbar) going to the unresolved open-charm worker.
    """
    from ..core import kinematics
    pdgs = tuple(int(value) for value in configuration["rows"][row_id]["pdg_ids"])
    mass = float(mass_gev)
    masses, charges = tuple(m0(pdg) for pdg in pdgs), tuple(charge(pdg) for pdg in pdgs)
    if len(pdgs) != 2:
        raise PortableModel1RealizerError(f"row {row_id!r} is not a two-body EventCalc row")
    if math.fsum(masses) > mass + 1.0e-12:
        raise PortableModel1RealizerError(f"row {row_id!r} is kinematically closed at {mass:g} GeV")
    local_seed = int(seed % 899_900_000) or 1
    if set(pdgs) <= STABLE_WITHOUT_PYTHIA:
        kinematics.seed(local_seed)
        generated = kinematics.two_body(mass, count, *masses, *pdgs, *charges, 0, 0)
        return generated.reshape(count, 2, 8)[:, :, :6].reshape(count, 12).tolist()
    from ..core.eventcalc import decay_rng, six_field
    binary = Path(configuration["runtime_binary"]).resolve()
    xmldoc = Path(configuration["runtime_xmldoc"]).resolve()
    if tuple(sorted(pdgs)) == (-4, 4):
        from ..dark_photon.open_charm import PythiaCharmEventBackend
        with RUNTIMES.use(('charm', binary, xmldoc), lambda: PythiaCharmEventBackend(
                binary, xmldoc, maximum_workers=2)) as backend:
            event = backend.unresolved_event(mass, local_seed)
        if event is None:
            raise PortableModel1RealizerError("charm worker returned no event")
    else:
        from ..core.event_worker import ExplicitPrimaryRuntime
        from ..core.seeds import derive_event_seeds
        with RUNTIMES.use(('primary', binary, xmldoc), lambda: ExplicitPrimaryRuntime(
                binary, xmldoc, maximum_condition_attempts=32, maximum_workers=8)) as runtime:
            event = runtime.generate(mass, pdgs, derive_event_seeds(local_seed, 0).provider)
    return [[float(value) for value in six_field(event, decay_rng(local_seed))]]
