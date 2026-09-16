"""Dark-photon absolute-width inputs and their ownership closure at one mass.

``data/dark-photon/inputs.json`` holds the contract scales, the source
components and two windows of supplied channels: the exclusive edge
[1.70, 2.00] GeV and the post-edge owners [2.00, 5.00] GeV.

residual = inclusive - external - resolved (every supplied channel is
subtracted once; a deficit beyond 16 ulp is an error, within it the residual
is clamped to zero), then split over the source components in proportion to
their relative weights (9:1:2 for the dark photon).
"""

from __future__ import annotations
import bisect
import dataclasses
import json
import math
from types import SimpleNamespace
from .. import DATA
from ..core.width_inputs import EXTERNAL_TREATMENT, RESOLVED_TREATMENT, InputSchemaError, InputSupportError, floating_tolerance
from .widths import unit_current_inclusive_uds_width


@dataclasses.dataclass(frozen=True, slots=True)
class ChannelWidth:
    channel_id: str
    treatment: str
    width_gev: float
    final_state_keys: tuple[str, ...]
    realizer_id: str | None


@dataclasses.dataclass(frozen=True, slots=True)
class ComponentResidual:
    component_id: str
    width_gev: float


@dataclasses.dataclass(frozen=True, slots=True)
class MassPointClosure:
    mass_gev: float
    inclusive_width_gev: float
    external_width_gev: float
    active_width_gev: float
    residual_width_gev: float
    closure_error_gev: float
    channels: tuple[ChannelWidth, ...]
    component_residuals: tuple[ComponentResidual, ...]


class ExclusiveChannel:
    """Supplied channel: linear width nodes (``masses``, ``widths``) above ``threshold_gev``.

    On the first ``direct_segment_count`` node intervals the width is instead
    max(0, p(m)) times the analytic inclusive width, with the C1 Hermite
    probability p from ``probability_nodes`` and ``probability_slopes_per_gev``;
    node widths are exact.
    """

    def __init__(self, data):
        self.__dict__.update(data)

    def linear_width_at(self, mass_gev: float, domain_gev: tuple[float, float]) -> float:
        """Zero at or below threshold, otherwise linear width interpolation without extrapolation."""

        minimum, maximum = domain_gev
        if not minimum <= mass_gev <= maximum:
            raise InputSupportError(f"exclusive mass {mass_gev} lies outside [{minimum}, {maximum}] GeV")
        if mass_gev <= self.threshold_gev:
            return 0.0
        masses, widths = list(self.masses), list(self.widths)
        if self.threshold_gev >= minimum:
            masses.insert(0, self.threshold_gev)
            widths.insert(0, 0.0)
        if not masses[0] <= mass_gev <= masses[-1]:
            raise InputSupportError(
                f"channel {self.channel_id} mass {mass_gev} lies outside [{masses[0]}, {masses[-1]}] GeV")
        index = bisect.bisect_left(masses, mass_gev)
        if masses[index] == mass_gev:
            return widths[index]
        fraction = (mass_gev - masses[index - 1]) / (masses[index] - masses[index - 1])
        return widths[index - 1] + fraction * (widths[index] - widths[index - 1])

    def width_at(self, mass_gev: float, domain_gev: tuple[float, float]) -> float:
        masses = self.masses

        if (mass_gev <= self.threshold_gev or not masses[0] <= mass_gev <= masses[-1]
                or not domain_gev[0] <= mass_gev <= domain_gev[1]):
            return self.linear_width_at(mass_gev, domain_gev)

        if mass_gev in masses:
            return self.widths[masses.index(mass_gev)]
        index = min(max(0, bisect.bisect_right(masses, mass_gev) - 1), len(masses) - 2)

        if index >= self.direct_segment_count:
            return self.linear_width_at(mass_gev, domain_gev)
        width = masses[index + 1] - masses[index]
        coordinate = (mass_gev - masses[index]) / width
        values, slopes = self.probability_nodes, self.probability_slopes_per_gev
        h00 = 2.0 * coordinate**3 - 3.0 * coordinate**2 + 1.0
        h10 = coordinate**3 - 2.0 * coordinate**2 + coordinate
        h01 = -2.0 * coordinate**3 + 3.0 * coordinate**2
        h11 = coordinate**3 - coordinate**2
        probability = (h00 * values[index] + h10 * width * slopes[index]
                       + h01 * values[index + 1] + h11 * width * slopes[index + 1])

        return max(0.0, probability) * unit_current_inclusive_uds_width(mass_gev)


class ClosureWindow:
    """Supplied ``channels`` and their realizer ``providers``, closed on ``support_gev``."""

    def __init__(self, data, components):
        self.support_gev = tuple(data["support_gev"])
        self.channels = tuple(ExclusiveChannel(item) for item in data["channels"])
        self.providers = data["providers"]
        self.components = components

    def closure_at(self, mass_gev: float) -> MassPointClosure:
        minimum, maximum = self.support_gev

        if not minimum <= mass_gev <= maximum:
            raise InputSupportError(f"closure mass {mass_gev} lies outside the window [{minimum}, {maximum}] GeV")
        mass = float(mass_gev)
        inclusive = unit_current_inclusive_uds_width(mass)
        channels = tuple(ChannelWidth(channel.channel_id, channel.treatment, channel.width_at(mass, self.support_gev),
                                      channel.final_state_keys, channel.realizer_id) for channel in self.channels)
        external = math.fsum(item.width_gev for item in channels if item.treatment == EXTERNAL_TREATMENT)
        resolved = math.fsum(item.width_gev for item in channels if item.treatment == RESOLVED_TREATMENT)
        subtotal = math.fsum((external, resolved))
        residual = inclusive - subtotal
        tolerance = floating_tolerance(inclusive, subtotal)

        if residual < -tolerance:
            raise InputSchemaError(
                f"supplied widths oversubscribe the inclusive width by {-residual} (floating tolerance {tolerance})")
        residual = max(residual, 0.0)
        active = math.fsum((resolved, residual))

        return MassPointClosure(mass, inclusive, external, active, residual, math.fsum((external, active)) - inclusive,
                                channels, _split_residual(self.components, residual))


def _split_residual(components, residual_width_gev: float) -> tuple[ComponentResidual, ...]:
    total = math.fsum(component.relative_weight for component in components)
    values = [residual_width_gev * (component.relative_weight / total) for component in components]
    correction_index = max(range(len(values)), key=values.__getitem__)

    for _ in range(4):
        correction = residual_width_gev - math.fsum(values)
        if correction == 0.0:
            break
        values[correction_index] += correction
    tolerance = floating_tolerance(residual_width_gev, *values)

    if (any(value < -tolerance for value in values)
            or not math.isclose(math.fsum(values), residual_width_gev, rel_tol=0.0, abs_tol=tolerance)):
        raise RuntimeError("component residual allocation failed floating closure")

    return tuple(ComponentResidual(component.component_id, 0.0 if value < 0.0 else value)
                 for component, value in zip(components, values))


def load_inputs() -> SimpleNamespace:
    """Contract scales, source components and the exclusive-edge and post-edge windows."""

    data = json.loads((DATA / "dark-photon" / "inputs.json").read_text())
    components = tuple(SimpleNamespace(**item) for item in data["components"])
    return SimpleNamespace(scales=SimpleNamespace(**data["scales"]), components=components,
                           exclusive_edge=ClosureWindow(data["exclusive_edge"], components),
                           post_edge=ClosureWindow(data["post_edge"], components))
