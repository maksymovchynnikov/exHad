"""Complete-event record (hadronic decay forest plus terminal particles) and its conservation checks."""

from __future__ import annotations
from collections import Counter
import math
from typing import NamedTuple
from .charge_completion import PionTopology

MOMENTUM_ABS_TOL_GEV = 1.0e-9

MOMENTUM_REL_TOL = 1.0e-10

MASS_SHELL_ABS_TOL_GEV2 = 1.0e-8

MASS_SHELL_REL_TOL = 1.0e-9


class EventModelError(ValueError):
    """Raised when an event violates the complete-event conservation contract."""


class ConservedCharges(NamedTuple):
    electric: int = 0
    baryon: int = 0
    lepton_e: int = 0
    lepton_mu: int = 0
    lepton_tau: int = 0

    def __add__(self, other):
        # Element-wise, never tuple concatenation.
        return ConservedCharges(*(a + b for a, b in zip(self, other)))


class ParticleDefinition(NamedTuple):
    mass_gev: float
    charges: ConservedCharges
    stable_for_output: bool


class FourMomentum(NamedTuple):
    px_gev: float
    py_gev: float
    pz_gev: float
    energy_gev: float


class StableParticle(NamedTuple):
    pdg_id: int
    momentum: FourMomentum


class AncestryNode(NamedTuple):
    node_id: int
    pdg_id: int
    parent_id: int | None
    terminal_index: int | None


class GeneratedEvent(NamedTuple):
    """Decay forest of the primary hadrons, node ids 0..n-1 in record order.

    Roots have no parent; every leaf is a terminal whose index binds it to
    ``stable_particles``.  The ownership cut and the pion topology rely on it.
    """

    nodes: tuple[AncestryNode, ...]
    stable_particles: tuple[StableParticle, ...]

    @property
    def primary_pdgs(self) -> tuple[int, ...]:
        return tuple(sorted(node.pdg_id for node in self.nodes if node.parent_id is None))


def direct_pion_topology(event: GeneratedEvent) -> PionTopology | None:
    """Pion counts when every node is a terminal primary pion, else None."""

    if any(node.parent_id is not None or node.terminal_index is None for node in event.nodes):
        return None
    pdgs = event.primary_pdgs
    if any(pdg != 111 and abs(pdg) != 211 for pdg in pdgs):
        return None
    counts = Counter(pdgs)
    return PionTopology(counts[211], counts[111], counts[-211])


def _close(observed: float, expected: float, *, absolute: float, relative: float) -> bool:
    return math.isclose(observed, expected, rel_tol=relative, abs_tol=absolute)


def validate_complete_event(event, *, parent_mass_gev, target_charges, catalog) -> None:
    """Raise EventModelError unless the event conserves everything.

    Check order decides retry versus failure for callers that retry on
    message text: catalog support, decay-node stability and charges, root
    charges, terminal stability and mass shells, four-momentum, total charges.
    """

    mass = float(parent_mass_gev)
    try:
        definitions = [catalog[node.pdg_id] for node in event.nodes]
    except KeyError as missing:
        raise EventModelError(
            f"PDG identifier {missing} is absent from the particle catalog") from None
    decays: dict[int, ConservedCharges] = {}
    for node in event.nodes:
        if node.parent_id is not None:
            decays[node.parent_id] = decays.get(node.parent_id, ConservedCharges()) + definitions[node.node_id].charges
    for parent_id in sorted(decays):
        if definitions[parent_id].stable_for_output:
            raise EventModelError("a decay node cannot be marked terminal-stable")
        if decays[parent_id] != definitions[parent_id].charges:
            raise EventModelError("ancestry graph violates conserved charges at a decay node")
    primary_charges = ConservedCharges()
    for node in event.nodes:
        if node.parent_id is None:
            primary_charges = primary_charges + definitions[node.node_id].charges
    if primary_charges != target_charges:
        raise EventModelError("ancestry primary roots do not match the target charges")
    px_values: list[float] = []
    py_values: list[float] = []
    pz_values: list[float] = []
    energy_values: list[float] = []
    total_charges = ConservedCharges()
    for particle in event.stable_particles:
        definition = catalog[particle.pdg_id]
        if not definition.stable_for_output:
            raise EventModelError(
                f"PDG {particle.pdg_id} is not stable in the output convention")
        momentum = particle.momentum
        observed_mass2 = (
            momentum.energy_gev * momentum.energy_gev
            - momentum.px_gev * momentum.px_gev
            - momentum.py_gev * momentum.py_gev
            - momentum.pz_gev * momentum.pz_gev
        )
        if not _close(
                observed_mass2, definition.mass_gev * definition.mass_gev,
                absolute=MASS_SHELL_ABS_TOL_GEV2,
                relative=MASS_SHELL_REL_TOL):
            raise EventModelError(
                f"PDG {particle.pdg_id} is outside the mass shell tolerance")
        px_values.append(momentum.px_gev)
        py_values.append(momentum.py_gev)
        pz_values.append(momentum.pz_gev)
        energy_values.append(momentum.energy_gev)
        total_charges = total_charges + definition.charges
    for values, label in ((px_values, "px"), (py_values, "py"), (pz_values, "pz")):
        if not _close(math.fsum(values), 0.0, absolute=MOMENTUM_ABS_TOL_GEV, relative=MOMENTUM_REL_TOL):
            raise EventModelError(f"event does not conserve {label}")
    if not _close(math.fsum(energy_values), mass, absolute=MOMENTUM_ABS_TOL_GEV, relative=MOMENTUM_REL_TOL):
        raise EventModelError("event does not conserve energy")
    if total_charges != target_charges:
        raise EventModelError(
            f"event charges {total_charges!r} do not match {target_charges!r}")
