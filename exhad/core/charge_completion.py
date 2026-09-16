"""Parameter-free statistical-isospin closure for continuum pion families.

The kernel traces the fixed-``(I, I3)`` projector over every independent
sequential coupling path of labeled isospin-one momentum slots.  It is a
maximum-ignorance closure ansatz inside a known source irrep, not a theorem
about an arbitrary exclusive amplitude.  Only the conditional charge orbit is
replaced; the weight of each source, primary-ancestry and pion-multiplicity
class is preserved.
"""

from __future__ import annotations
import dataclasses
from fractions import Fraction


class ChargeCompletionError(ValueError):
    """Raised when a charge orbit cannot be completed without guessing."""


@dataclasses.dataclass(frozen=True, slots=True, order=True)
class PionTopology:
    """Counts of final ``pi+``, ``pi0``, and ``pi-`` particles."""

    n_plus: int
    n_zero: int
    n_minus: int

    @property
    def multiplicity(self) -> int:
        return self.n_plus + self.n_zero + self.n_minus

    @property
    def charge(self) -> int:
        return self.n_plus - self.n_minus


def _cg_squared_j_times_one(
        previous_i: int, previous_i3: int, pion_i3: int,
        total_i: int) -> Fraction:
    """Exact squared integer-isospin Clebsch--Gordan coefficient."""

    total_i3 = previous_i3 + pion_i3
    if total_i < 0 or abs(total_i3) > total_i or abs(previous_i3) > previous_i:
        return Fraction(0)
    if total_i == previous_i + 1:
        if pion_i3 == +1:
            numerator = ((previous_i + previous_i3 + 1)
                         * (previous_i + previous_i3 + 2))
            denominator = 2 * (previous_i + 1) * (2 * previous_i + 1)
        elif pion_i3 == 0:
            numerator = ((previous_i - previous_i3 + 1)
                         * (previous_i + previous_i3 + 1))
            denominator = (previous_i + 1) * (2 * previous_i + 1)
        else:
            numerator = ((previous_i - previous_i3 + 1)
                         * (previous_i - previous_i3 + 2))
            denominator = 2 * (previous_i + 1) * (2 * previous_i + 1)
    elif total_i == previous_i and previous_i > 0:
        denominator = previous_i * (previous_i + 1)
        if pion_i3 == +1:
            numerator = ((previous_i - previous_i3)
                         * (previous_i + previous_i3 + 1))
            denominator *= 2
        elif pion_i3 == 0:
            numerator = previous_i3 * previous_i3
        else:
            numerator = ((previous_i + previous_i3)
                         * (previous_i - previous_i3 + 1))
            denominator *= 2
    elif total_i == previous_i - 1 and previous_i > 0:
        if pion_i3 == +1:
            numerator = ((previous_i - previous_i3 - 1)
                         * (previous_i - previous_i3))
            denominator = 2 * previous_i * (2 * previous_i + 1)
        elif pion_i3 == 0:
            numerator = ((previous_i - previous_i3)
                         * (previous_i + previous_i3))
            denominator = previous_i * (2 * previous_i + 1)
        else:
            numerator = ((previous_i + previous_i3 - 1)
                         * (previous_i + previous_i3))
            denominator = 2 * previous_i * (2 * previous_i + 1)
    else:
        return Fraction(0)
    if numerator <= 0:
        return Fraction(0)
    return Fraction(numerator, denominator)


def statistical_isospin_probabilities(
        n_pions: int, total_isospin2: int, isospin3_2: int = 0,
        ) -> dict[PionTopology, Fraction]:
    """Return the exact statistical charge orbit in a fixed source irrep.

    Isospin inputs use doubled-integer notation; a pion system has integer
    total isospin, so odd doubled values are rejected rather than rounded.
    """

    if (n_pions < 1 or total_isospin2 % 2 or isospin3_2 % 2
            or abs(isospin3_2) > total_isospin2 or total_isospin2 > 2 * n_pions):
        raise ChargeCompletionError(
            "pion systems require integer I and I3 inside an irrep reachable "
            "at this multiplicity")
    total_i = total_isospin2 // 2
    total_i3 = isospin3_2 // 2

    # State key: intermediate I, I3, n+, n0, n-.  The exact weights sum over
    # every orthonormal sequential-coupling path and every charge ordering.
    states: dict[tuple[int, int, int, int, int], Fraction] = {}
    for pion_i3, counts in (
            (+1, (1, 0, 0)), (0, (0, 1, 0)), (-1, (0, 0, 1))):
        states[(1, pion_i3, *counts)] = Fraction(1)

    for _ in range(1, n_pions):
        updated: dict[tuple[int, int, int, int, int], Fraction] = {}
        for state, weight in states.items():
            previous_i, previous_i3, n_plus, n_zero, n_minus = state
            for pion_i3, delta in (
                    (+1, (1, 0, 0)), (0, (0, 1, 0)), (-1, (0, 0, 1))):
                for new_i in (previous_i - 1, previous_i, previous_i + 1):
                    coefficient = _cg_squared_j_times_one(
                        previous_i, previous_i3, pion_i3, new_i)
                    if not coefficient:
                        continue
                    key = (
                        new_i,
                        previous_i3 + pion_i3,
                        n_plus + delta[0],
                        n_zero + delta[1],
                        n_minus + delta[2],
                    )
                    updated[key] = updated.get(key, Fraction(0)) + \
                        weight * coefficient
        states = updated

    topology_weights: dict[PionTopology, Fraction] = {}
    for state, weight in states.items():
        isospin, isospin3, n_plus, n_zero, n_minus = state
        if isospin == total_i and isospin3 == total_i3:
            topology = PionTopology(n_plus, n_zero, n_minus)
            topology_weights[topology] = \
                topology_weights.get(topology, Fraction(0)) + weight
    normalization = sum(topology_weights.values(), Fraction(0))
    if normalization <= 0:
        raise ChargeCompletionError(
            "requested source irrep has no support at this pion multiplicity")
    return {
        topology: weight / normalization
        for topology, weight in sorted(topology_weights.items())
        if weight > 0
    }
