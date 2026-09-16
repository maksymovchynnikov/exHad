"""Dark-photon light-uds event measure.

Three regimes on the contract scales.  At or below the fragmentation edge
(2.0 GeV) the exclusive-edge closure is sampled: external channels, resolved
exclusive channels and per-component residuals.  Up to the handoff start
(4.0 GeV) the frozen EM authority supplies the fitted response widths plus the
residual, split by relative component weight.  Up to the stock endpoint
(5.0 GeV) only the residual kernel blends to stock Pythia with
w(m) = 1 - 3x^2 + 2x^3.

Every event draws, with fixed splitmix64 seed domains, an owner (external
channel versus the active sector, by width over the inclusive width) and a
leaf inside the active sector.  Leaf order is part of the seeded stream.
Exclusive leaves must return one of their own ownership keys; continuum
leaves may not return an exclusively owned key (below 2 GeV all exclusive-edge
keys, above it the post-edge keys and the fitted-route keys).
"""

from __future__ import annotations
from fractions import Fraction
import math
from typing import NamedTuple
from ..core.charge_completion import statistical_isospin_probabilities
from ..core.event_model import ConservedCharges, MOMENTUM_ABS_TOL_GEV, direct_pion_topology, validate_complete_event
from ..core.event_worker import EventRequest
from ..core.ownership import ownership_key
from ..core.seeds import derive_event_seeds, derive_handoff_seed, derive_provider_attempt_seed, derive_seam_seed
from ..core.width_inputs import EXTERNAL_TREATMENT, final_state_key_from_pdgs

NATIVE_PION_MAX_ATTEMPTS = 64


class ConstructedEventError(ValueError):
    """Raised when a dark-photon plan or generated event violates its contract."""


class ProviderSupportError(ConstructedEventError):
    """Raised when provider output violates final-state ownership."""


class NativeChargeCompletionExhausted(ConstructedEventError):
    """Raised when a fixed native pion target exhausts its retry budget."""


class PostEdgeRatePoint(NamedTuple):
    """Absolute active-sector widths supplied by the frozen EM authority."""

    mass_gev: float
    active_width_gev: float
    resolved_widths_gev: dict
    residual_width_gev: float


class Leaf(NamedTuple):
    kind: str  # external, resolved, residual, stock, or the owner-draw token active
    owner_id: str  # channel_id of an exclusive leaf, component_id of a continuum leaf
    provider_id: str | None
    width_gev: float
    realizer_id: str | None = None
    final_state_keys: tuple[str, ...] = ()


class Plan(NamedTuple):
    owners: tuple[Leaf, ...]  # external leaves by channel_id, then the active token
    inclusive_width_gev: float
    active: tuple[Leaf, ...]
    active_width_gev: float
    forbidden: frozenset[str]
    responses: tuple[Leaf, ...] = ()
    residuals: tuple[Leaf, ...] = ()
    stock: tuple[Leaf, ...] = ()
    constructed_weight: float = 1.0
    handoff: bool = False


def _unit(seed: int) -> float:
    # The top 53 bits form a reproducible binary fraction strictly below one.
    return (seed >> 11) / float(1 << 53)


def _weighted_select(items: tuple[Leaf, ...], total: float, seed: int) -> Leaf:
    if not items:
        raise ConstructedEventError("cannot sample an empty event measure")
    target = _unit(seed) * total
    cumulative = 0.0

    for item in items:
        cumulative += item.width_gev
        if target < cumulative:
            return item

    return items[-1]


def _by_owner(leaves) -> tuple[Leaf, ...]:
    return tuple(sorted(leaves, key=lambda leaf: leaf.owner_id))


def _draw(plan: Plan, seeds) -> Leaf:
    owner = _weighted_select(plan.owners, plan.inclusive_width_gev, seeds.owner)

    if owner.kind != "active":
        return owner

    if not plan.handoff:
        return _weighted_select(plan.active, plan.active_width_gev, seeds.leaf)
    response = math.fsum(leaf.width_gev for leaf in plan.responses)
    residual = math.fsum(leaf.width_gev for leaf in plan.residuals)
    weight = plan.constructed_weight
    total = math.fsum((response, weight * residual))

    if _unit(derive_handoff_seed(seeds)) >= total / plan.active_width_gev:
        if (1.0 - weight) * residual <= 0.0:
            raise ConstructedEventError("stock residual sector is empty")
        return _weighted_select(
            plan.stock, math.fsum(leaf.width_gev for leaf in plan.stock), seeds.leaf)
    # At 4 GeV replay the lower-regime draw, not a repartition with a new seed.

    if weight == 1.0:
        return _weighted_select(plan.active, plan.active_width_gev, seeds.leaf)

    if total <= 0.0:
        raise ConstructedEventError("constructed handoff sector is empty")

    if response > 0.0 and _unit(derive_seam_seed(seeds)) < response / total:
        return _weighted_select(plan.responses, response, seeds.leaf)

    if weight <= 0.0 or residual <= 0.0:
        raise ConstructedEventError("constructed residual kernel is empty")

    return _weighted_select(plan.residuals, residual, seeds.leaf)


def _constructive_weight(scales, mass_gev: float) -> float:
    """Constructed share 1 - 3x^2 + 2x^3 across the handoff to stock Pythia."""
    if mass_gev <= scales.handoff_start_gev:
        return 1.0
    if mass_gev >= scales.stock_endpoint_gev:
        return 0.0
    x = (mass_gev - scales.handoff_start_gev) / (scales.stock_endpoint_gev - scales.handoff_start_gev)
    return 1.0 - 3.0 * x * x + 2.0 * x * x * x


def _direct_pion_topology(component, event):
    """Topology of a direct-primary pure-pion event the component's G-parity admits, else None."""

    topology = direct_pion_topology(event)
    if topology is None:
        return None
    if topology.charge * 2 != component.isospin3_2:
        raise ConstructedEventError(
            "validated event charge is incompatible with source isospin3")
    policy, count = component.pion_multiplicity_policy, topology.multiplicity
    if policy == "all" or (policy == "even" and count % 2 == 0) or (policy == "odd" and count % 2 == 1):
        return topology
    return None


class ConstructedEventMeasure:
    """Frozen dark-photon inputs bound to a dict of provider callables."""

    def __init__(self, inputs, authority, catalog, providers, reference_provider_id, runtimes):
        self.inputs = inputs
        self.authority = authority
        self.catalog = catalog
        self.providers = providers
        self.reference_provider_id = reference_provider_id
        self._runtimes = runtimes
        self._routes = {route.response_block_id: route for route in authority.routes}
        self.components = {component.component_id: component for component in inputs.components}
        self.inclusive_forbidden_keys = frozenset(
            key for channel in inputs.exclusive_edge.channels for key in channel.final_state_keys)
        # Also the stable-light veto of open-charm events.
        self.post_edge_forbidden_keys = frozenset(
            key for channel in inputs.post_edge.channels for key in channel.final_state_keys).union(
                key for route in authority.routes for key in route.final_state_keys)
        self._plan_cache = (None, None)

    def close(self) -> None:
        for runtime in self._runtimes:
            runtime.close()

    def _channel_leaves(self, closure, providers):
        external, resolved = [], []

        for channel in closure.channels:
            if channel.width_gev <= 0.0:
                continue
            if channel.realizer_id is None:
                raise ConstructedEventError("positive channel is missing its realizer")
            leaf = Leaf(channel.treatment, channel.channel_id, providers[channel.realizer_id],
                        channel.width_gev, channel.realizer_id, tuple(channel.final_state_keys))
            (external if channel.treatment == EXTERNAL_TREATMENT else resolved).append(leaf)

        return _by_owner(external), _by_owner(resolved)

    def _plan(self, mass: float) -> Plan:
        scales = self.inputs.scales

        if mass <= scales.fragmentation_edge_gev:
            closure = self.inputs.exclusive_edge.closure_at(mass)
            external, resolved = self._channel_leaves(closure, self.inputs.exclusive_edge.providers)
            residuals = _by_owner(
                Leaf("residual", item.component_id, self.reference_provider_id, item.width_gev)
                for item in closure.component_residuals if item.width_gev > 0.0)
            owner_leaves = external
            if closure.active_width_gev > 0.0:
                owner_leaves += (Leaf("active", "", None, closure.active_width_gev),)
            # Exclusive plans sort residuals before resolved channels.
            return Plan(owner_leaves, closure.inclusive_width_gev, residuals + resolved,
                        closure.active_width_gev, self.inclusive_forbidden_keys)
        closure = self.inputs.post_edge.closure_at(mass)
        external, resolved = self._channel_leaves(closure, self.inputs.post_edge.providers)

        if resolved:
            raise ConstructedEventError("post-edge closure contains a non-external channel")

        if closure.active_width_gev <= 0.0:
            return Plan(external, closure.inclusive_width_gev, (), 0.0, self.post_edge_forbidden_keys)
        point = self.authority.rate_point_at(
            mass, inclusive_width_gev=closure.inclusive_width_gev,
            external_width_gev=closure.external_width_gev)
        responses = _by_owner(
            Leaf("resolved", route.channel_id, route.provider_id, width, route.realizer_id,
                 tuple(route.final_state_keys))
            for route, width in ((self._routes[block], width)
                                 for block, width in point.resolved_widths_gev.items())
            if width > 0.0)
        weights = tuple((item.component_id, float(item.relative_weight))
                        for item in self.inputs.components)
        weight_total = math.fsum(weight for _, weight in weights)
        residuals = _by_owner(
            leaf for leaf in (
                Leaf("residual", component_id, self.reference_provider_id,
                     point.residual_width_gev * weight / weight_total)
                for component_id, weight in weights)
            if leaf.width_gev > 0.0)
        handoff = mass > scales.handoff_start_gev
        stock = _by_owner(Leaf("stock", component_id, self.reference_provider_id, weight)
                          for component_id, weight in weights) if handoff else ()
        # Fitted plans sort responses before residuals.

        return Plan(external + (Leaf("active", "", None, closure.active_width_gev),),
                    closure.inclusive_width_gev, responses + residuals, point.active_width_gev,
                    self.post_edge_forbidden_keys, responses, residuals, stock,
                    _constructive_weight(scales, mass) if handoff else 1.0, handoff)

    def _request(self, mass, leaf, forbidden, seed, condition=None):
        continuum = None if leaf.kind in ("external", "resolved") else leaf.owner_id
        return EventRequest(mass, seed, continuum, leaf.realizer_id, leaf.final_state_keys, forbidden, condition)

    def _validate(self, mass, event, leaf, forbidden) -> None:
        validate_complete_event(
            event, parent_mass_gev=mass, target_charges=ConservedCharges(), catalog=self.catalog)
        exclusive = leaf.kind in ("external", "resolved")
        # A branch outside the quasi-stable convention (key None) is fatal for
        # an exclusive leaf and channel-level `other` for the continuum.
        key = ownership_key(event)

        if exclusive and key not in leaf.final_state_keys:
            raise ProviderSupportError(
                "exclusive provider emitted an ownership state outside its support")

        if not exclusive and key in forbidden:
            raise ProviderSupportError("residual provider emitted an exclusively owned state")

    def generate(self, mass_gev: float, event_index: int, master_seed: int):
        mass = float(mass_gev)

        if self._plan_cache[0] != mass:
            self._plan_cache = (mass, self._plan(mass))
        plan = self._plan_cache[1]
        seeds = derive_event_seeds(master_seed, event_index)
        leaf = _draw(plan, seeds)
        forbidden = frozenset() if leaf.kind in ("external", "resolved") else plan.forbidden
        provider = self.providers[leaf.provider_id]
        event = provider(self._request(mass, leaf, forbidden, seeds.provider))
        self._validate(mass, event, leaf, forbidden)

        if leaf.kind != "residual":
            return event
        component = self.components[leaf.owner_id]
        observed = _direct_pion_topology(component, event)

        if observed is None:
            return event
        # Native statistical-isospin completion: condition the charge orbit on
        # unowned, kinematically open topologies, draw one exactly, and retry
        # the same route until it produces that topology.
        allowed = []

        for topology, probability in statistical_isospin_probabilities(
                observed.multiplicity, component.total_isospin2, component.isospin3_2).items():
            pdgs = (211,) * topology.n_plus + (111,) * topology.n_zero + (-211,) * topology.n_minus
            # Direct primary pions are already at the ownership cut.
            if final_state_key_from_pdgs(pdgs) in forbidden:
                continue
            # A topology at rest-mass threshold has zero phase-space measure.
            if math.fsum(self.catalog[pdg].mass_gev for pdg in pdgs) >= mass - MOMENTUM_ABS_TOL_GEV:
                continue
            allowed.append((topology, probability))
        normalization = sum((probability for _, probability in allowed), Fraction(0))

        if normalization <= 0:
            raise ProviderSupportError("no unowned accessible pion topology remains in the residual")
        allowed.sort()
        threshold, cumulative, target = Fraction(seeds.topology, 1 << 64), Fraction(0), allowed[-1][0]

        for topology, probability in allowed:
            cumulative += probability / normalization
            if threshold < cumulative:
                target = topology
                break

        if target == observed:
            return event

        for attempt in range(1, NATIVE_PION_MAX_ATTEMPTS):
            event = provider(self._request(
                mass, leaf, forbidden, derive_provider_attempt_seed(seeds, attempt), target))
            self._validate(mass, event, leaf, forbidden)
            if _direct_pion_topology(component, event) == target:
                return event
        raise NativeChargeCompletionExhausted(
            "native pion target exhausted its fixed same-route retry budget")
