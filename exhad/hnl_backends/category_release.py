"""C2 release of a value-and-slope matched boundary composition."""

from __future__ import annotations
import dataclasses
import math
from ..core.width_inputs import floating_tolerance


class CategoryReleaseError(ValueError):
    """Raised when a boundary release loses its probability simplex."""


@dataclasses.dataclass(frozen=True, slots=True)
class SimplexRelease:
    """Positive C2 release of a value-and-slope matched composition.

    The retained weight is the quintic smootherstep
    ``R(x) = 1 - 10 x**3 + 15 x**4 - 6 x**5``.  The conditional composition
    is continued independently of ``R``: every decreasing component ``i``
    relaxes its boundary derivative as ``c_i'(y) = s_i max(1 - y/d_i, 0)**2``
    with ``d_i = min(L, 3 p_i / -s_i)``, and increasing components share the
    aggregate decrease in proportion to their slopes, which keeps exact
    simplex closure.  ``R`` times the composition preserves the boundary value
    and derivative and vanishes with two derivatives at the endpoint.
    """

    start_gev: float
    endpoint_gev: float
    boundary_probabilities: tuple[float, ...]
    boundary_slopes_per_gev: tuple[float, ...]
    relaxation_lengths_gev: tuple[float, ...] = dataclasses.field(init=False)
    _negative_slope_sum_per_gev: float = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        start, endpoint = float(self.start_gev), float(self.endpoint_gev)
        probabilities = tuple(float(value) for value in self.boundary_probabilities)
        slopes = tuple(float(value) for value in self.boundary_slopes_per_gev)
        length = endpoint - start
        relaxation_lengths = []
        negative_slope_sum = 0.0

        for probability, slope in zip(probabilities, slopes):
            if slope < 0.0:
                if probability == 0.0:
                    raise CategoryReleaseError(
                        "a zero boundary probability cannot have a negative slope")
                relaxation_lengths.append(min(length, 3.0 * probability / -slope))
                negative_slope_sum -= slope
            else:
                relaxation_lengths.append(length)
        object.__setattr__(self, "start_gev", start)
        object.__setattr__(self, "endpoint_gev", endpoint)
        object.__setattr__(self, "boundary_probabilities", probabilities)
        object.__setattr__(self, "boundary_slopes_per_gev", slopes)
        object.__setattr__(self, "relaxation_lengths_gev", tuple(relaxation_lengths))
        object.__setattr__(self, "_negative_slope_sum_per_gev", negative_slope_sum)

    def retained_weight(self, mass_gev: float) -> float:
        """Return the C2 probability still assigned to boundary owners."""

        mass = float(mass_gev)
        if mass <= self.start_gev:
            return 1.0
        if mass >= self.endpoint_gev:
            return 0.0
        x = (mass - self.start_gev) / (self.endpoint_gev - self.start_gev)
        return 1.0 - 10.0 * x**3 + 15.0 * x**4 - 6.0 * x**5

    def conditional_composition_at(self, mass_gev: float) -> tuple[float, ...]:
        """Return the closed conditional composition before applying R."""

        mass = float(mass_gev)
        if mass <= self.start_gev:
            return self.boundary_probabilities
        distance = min(mass, self.endpoint_gev) - self.start_gev
        decreasing_integrals = []
        for slope, relaxation_length in zip(self.boundary_slopes_per_gev, self.relaxation_lengths_gev):
            # Integral of max(1 - y/d, 0)**2 from zero to the distance.
            reached = min(distance, relaxation_length)
            ratio = reached / relaxation_length
            decreasing_integrals.append(
                reached * (1.0 - ratio + ratio * ratio / 3.0) if slope < 0.0 else 0.0)
        aggregate_decrease = math.fsum(
            -slope * integral
            for slope, integral in zip(self.boundary_slopes_per_gev, decreasing_integrals)
            if slope < 0.0)
        composition = []
        for probability, slope, integral in zip(
                self.boundary_probabilities, self.boundary_slopes_per_gev, decreasing_integrals):
            if slope < 0.0:
                value = probability + slope * integral
            elif slope > 0.0:
                value = probability + slope * aggregate_decrease / self._negative_slope_sum_per_gev
            else:
                value = probability
            if value < 0.0 and value >= -floating_tolerance(value, 0.0, 1.0):
                value = 0.0
            composition.append(value)
        if any(value < 0.0 for value in composition):
            raise CategoryReleaseError("simplex continuation produced a negative probability")
        return tuple(composition)

    def retained_probabilities_at(self, mass_gev: float) -> tuple[float, ...]:
        """Return nonnegative retained probabilities for each boundary row."""

        mass = float(mass_gev)
        if mass <= self.start_gev:
            return self.boundary_probabilities
        if mass >= self.endpoint_gev:
            return tuple(0.0 for _ in self.boundary_probabilities)
        retained = self.retained_weight(mass)
        result = tuple(retained * probability for probability in self.conditional_composition_at(mass))
        tolerance = floating_tolerance(*result, 0.0, 1.0)
        result = tuple(0.0 if value < 0.0 and value >= -tolerance else value for value in result)
        if any(value < 0.0 for value in result):
            raise CategoryReleaseError("simplex release produced a negative probability")
        return result
