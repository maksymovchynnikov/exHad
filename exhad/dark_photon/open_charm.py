"""Open-charm events of the dark-photon total-EM measure.

The rate authority is :mod:`exhad.dark_photon.charm_rates`.  Each measured
family with positive cross section is realized from its declared on-shell
``D``, ``D*``, ``Ds*`` or ``Lambda_c`` primaries plus ordinary Pythia decays;
the inclusive remainder is an exact primary ``c cbar`` string.  Both are
conditioned on the complement of the stable-light ownership keys already
owned by the frozen light response, so the inclusive charm width is unchanged
and no stable final state is owned twice.
"""

from __future__ import annotations
from collections import Counter
import math
from typing import NamedTuple
from ..core.event_model import ConservedCharges, validate_complete_event
from ..core.event_worker import EventWorker, WorkerPool, particle_catalog_from_pythia_xml
from ..core.ownership import ownership_key
from ..core.pdg import m0
from ..core.seeds import splitmix64
from .charm_rates import build_open_charm_owner

_OWNER_DRAW_TAG = 0x1A2E6F97753E8B45

_FAMILY_DRAW_TAG = 0xAB41256F6D729586  # fixed seed domain of the charm-family draw

_MODE_DRAW_TAG = 0xC84B1D939A7E503B

_ATTEMPT_TAG = 0x799731DAE16B2543


class CharmEventError(ValueError):
    """The charm event measure is incomplete or cannot be sampled."""


class CharmPrimaryMode(NamedTuple):
    """One fixed on-shell primary configuration inside a measured family."""

    primary_pdgs: tuple[int, ...]
    relative_weight: float

_DECLARED_PRIMARY_PDGS = {
    "ddbar": ((421, -421), (411, -411)),
    "dstar_d_charged_cc": ((413, -411), (-413, 411)),
    "dstar_charged_pair": ((413, -413),),
    "dsstar_pair": ((433, -433),),
    "lambdac_pair": ((4122, -4122),),
    "pipi_dd_charged": ((211, -211, 411, -411),),
}


def charm_primary_modes_at(channel_id: str, mass_gev: float) -> tuple[CharmPrimaryMode, ...]:
    """Return the parameter-free on-shell primary response of one measured family.

    A 1-- current produces two pseudoscalars in a P wave; equal isospin
    amplitudes give D Dbar weights p^3, including the charged/neutral
    threshold offset.  Charge-conjugate pairs have equal weights.
    """

    mass = float(mass_gev)
    modes = []
    for pdgs in _DECLARED_PRIMARY_PDGS[channel_id]:
        if mass < math.fsum(m0(pdg) for pdg in pdgs):
            continue
        weight = 1.0
        if channel_id == "ddbar":
            first, second = (m0(pdg) for pdg in pdgs)
            radicand = (mass * mass - (first + second) ** 2) * (mass * mass - (first - second) ** 2)
            weight = (math.sqrt(radicand) / (2.0 * mass) if radicand > 0.0 else 0.0) ** 3
        if weight > 0.0:
            modes.append(CharmPrimaryMode(pdgs, weight))
    if not modes:
        raise CharmEventError(
            f"measured charm family {channel_id!r} has no on-shell primary at {mass:.12g} GeV")
    return tuple(modes)


def _unit_interval(seed: int, tag: int) -> float:
    return (splitmix64(seed ^ tag) >> 11) / float(1 << 53)


def _weighted_choice(items, weights, unit: float):
    total = math.fsum(weights)

    if not items or total <= 0.0:
        raise CharmEventError("cannot draw from an empty charm measure")
    target = unit * total
    cumulative = 0.0

    for item, weight in zip(items, weights, strict=True):
        cumulative += weight
        if target < cumulative:
            return item

    return items[-1]


class PythiaCharmEventBackend:
    """Persistent Pythia realization of on-shell (pole-mass) and exact-cc charm events."""

    def __init__(self, binary, xmldoc, *, maximum_workers=12):
        self.pool = WorkerPool(maximum_workers)
        self.binary, self.xmldoc = binary, xmldoc

    def close(self) -> None:
        self.pool.close()

    def resolved_event(self, mass_gev, primary_pdgs, seed):
        pdgs = tuple(primary_pdgs)
        return self.pool.request(("resolved", pdgs),
                            lambda: EventWorker(self.binary, self.xmldoc, inject=pdgs), mass_gev, seed)

    def unresolved_event(self, mass_gev, seed):
        return self.pool.request(("unresolved-cc",),
                            lambda: EventWorker(self.binary, self.xmldoc, flavor="c"), mass_gev, seed)


class CharmEventGenerator(NamedTuple):
    owner: object
    backend: PythiaCharmEventBackend
    catalog: object

    def close(self) -> None:
        self.backend.close()


def build_pythia_charm_event_generator(*, binary, xmldoc) -> CharmEventGenerator:
    """Build the production charm generator (no fit parameters)."""

    return CharmEventGenerator(
        build_open_charm_owner(), PythiaCharmEventBackend(binary, xmldoc),
        particle_catalog_from_pythia_xml(xmldoc))


def _charm_event(generator, point, leaves, event_seed, veto):
    # The family has its own uniform: reusing the light/charm owner draw would
    # confine charm events to cumulative leaf probability above p_light.
    modes = _weighted_choice(
        tuple(modes for _, modes in leaves), tuple(probability for probability, _ in leaves),
        _unit_interval(event_seed, _FAMILY_DRAW_TAG))

    for attempt in range(1024):
        attempt_seed = splitmix64(event_seed ^ splitmix64(_ATTEMPT_TAG + attempt))
        mode = None
        if modes is not None:
            mode = _weighted_choice(modes, tuple(item.relative_weight for item in modes),
                                    _unit_interval(attempt_seed, _MODE_DRAW_TAG))
            event = generator.backend.resolved_event(point.mass_gev, mode.primary_pdgs, attempt_seed)
        else:
            event = generator.backend.unresolved_event(point.mass_gev, attempt_seed)
        # Leptonic/semileptonic charm decays lie outside the light ownership
        # convention (key None), hence never in the veto.
        if event is None or ownership_key(event) in veto:
            continue
        validate_complete_event(event, parent_mass_gev=point.mass_gev,
                                target_charges=ConservedCharges(), catalog=generator.catalog)
        if mode is not None and Counter(event.primary_pdgs) != Counter(mode.primary_pdgs):
            raise CharmEventError(
                "resolved charm event roots differ from the selected on-shell primary configuration")
        return event
    raise CharmEventError(
        "charm event generator exhausted its stable-light complement; "
        "the veto may remove all support from a nonzero charm owner")


def generate_total_em_dark_photon_events(model, charm_generator, mass_gev, event_count, master_seed):
    """Generate the disjoint light-uds plus open-charm measure; light only below the charm domain."""

    mass = float(mass_gev)
    if mass < charm_generator.owner.card.domain_gev[0]:
        return [model.generate(mass, index, master_seed) for index in range(event_count)]
    light = model.inputs.post_edge.closure_at(mass).inclusive_width_gev
    point = charm_generator.owner.closure_at(mass)
    total = math.fsum((light, point.inclusive_charm_width))
    light_probability = light / total
    charm_probability = point.inclusive_charm_width / total
    leaves = []  # (probability, primary modes or None for the unresolved string)
    if point.inclusive_charm_width > 0.0:
        leaves = [(item.width / point.inclusive_charm_width, charm_primary_modes_at(item.channel_id, point.mass_gev))
                  for item in point.resolved if item.sigma_nb > 0.0]
        if point.residual_width > 0.0:
            leaves.append((point.residual_width / point.inclusive_charm_width, None))
    events = []
    for index in range(event_count):
        event_seed = splitmix64(master_seed ^ splitmix64(index))
        if charm_probability > 0.0 and _unit_interval(event_seed, _OWNER_DRAW_TAG) >= light_probability:
            events.append(_charm_event(charm_generator, point, leaves, event_seed,
                                       model.post_edge_forbidden_keys))
        else:
            events.append(model.generate(mass, index, master_seed))
    return events
