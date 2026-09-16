"""Portable Model-1 family probabilities inside the active fragmentation pool.

Each named family carries P_F = B_F (m_split/m)^{2p} S_F(m): the boundary
probability, the common two-parton power law and a log-log PCHIP saturation
factor. The Pythia proposal q_F is the frozen mixed family response, and the
rejection weight is P_F/q_F. Modes listed in ``removed_external_modes`` stay
with the outer owners that supplied their rates.
"""

from __future__ import annotations
from collections import Counter
import math
from types import SimpleNamespace

BOUNDARY_CONDITIONED_RESPONSE = "boundary-conditioned"
VARIATIONS = frozenset({"central", "saturation-off", "p-low", "p-high"})
PIONS = frozenset({-211, 111, 211})
KAONS = frozenset({-321, -311, 130, 310, 311, 321})


class PortableModel1Error(ValueError):
    """A portable Model-1 evaluation or sample failed closed."""


def canonical_channel_key(pdgs) -> str:
    counts = Counter(int(pdg) for pdg in pdgs)
    return " ".join(f"{pdg}:{counts[pdg]}" for pdg in sorted(counts))


def classify_channel(classifier_id: str, pdgs) -> str:
    """Family of a canonical ownership state (first matching rule wins).

    The rules are mirrored by cpp/include/portable_rejection.h and must stay
    in lockstep with it.
    """
    state = sorted(int(pdg) for pdg in pdgs)
    n = len(state)
    pions = sum(pdg in PIONS for pdg in state)
    kaons = sum(pdg in KAONS for pdg in state)
    etas = sum(pdg in (221, 331) for pdg in state)
    eta221 = state.count(221)
    only_pions = n > 0 and pions == n
    baryon_pair = any(pdg >= 1000 for pdg in state) and any(pdg <= -1000 for pdg in state)
    vector = (
        ("exact-pi0-gamma", state == [22, 111]),
        ("exact-2pi", only_pions and n == 2), ("exact-3pi", only_pions and n == 3),
        ("exact-4pi", only_pions and n == 4), ("exact-6pi", only_pions and n == 6),
        ("remainder", only_pions),
        ("exact-kk", n == 2 and kaons == 2),
        ("exact-kkpi", n == 3 and kaons == 2 and pions == 1),
        ("exact-kkpipi", n == 4 and kaons == 2 and pions == 2),
        ("exact-nucleon-pair", state in ([-2212, 2212], [-2112, 2112])),
    )
    rules = {
        "b-l-resolved-families": (
            ("exact-5pi", only_pions and n == 5),
            ("eta-three-pion", eta221 == 1 and n == 4 and pions == 3),
            ("eta-kaon-pair", eta221 == 1 and n == 3 and kaons == 2),
        ) + vector,
        "alp-fermion-exact-families": (
            ("baryon-pair", any(abs(pdg) >= 1000 for pdg in state)),
            ("kkpi", n == 3 and kaons == 2 and pions == 1),
            ("kkpipi", kaons > 0 and pions >= 2), ("remainder", kaons > 0),
            ("eta-pipi", etas == 1 and pions == 2 and n == 3), ("remainder", etas > 0),
            ("three-pion", pions == 3 and n == 3), ("pipi-gamma", state == [-211, 22, 211]),
            ("remainder", pions == 3 or (pions == 2 and 22 in state)),
            ("four-pion", pions == 4), ("fiveplus-pion", pions >= 5),
        ),
        "scalar-families": (
            ("pipi", n == 2 and pions == 2), ("kk", n == 2 and kaons == 2),
            ("four-pion", n == 4 and pions == 4), ("baryon-pair", baryon_pair),
            ("eta-multihadron", etas > 0), ("kaon-multihadron", kaons > 0),
            ("higher-multipion", n >= 5 and pions == n),
        ),
        "hnl-fixed-w-families": (
            ("nucleon-pair", baryon_pair),
            ("two-pion", only_pions and n == 2), ("three-pion", only_pions and n == 3),
            ("four-pion", only_pions and n == 4), ("five-plus", only_pions and n == 5),
            ("remaining", only_pions),
            ("kaon-pair", n == 2 and kaons == 2),
            ("kaon-pair-pion", n == 3 and kaons == 2 and pions == 1),
            ("eta-pion-pion", n == 3 and eta221 == 1 and pions == 2),
        ),
    }[classifier_id]
    default = "remaining" if classifier_id == "hnl-fixed-w-families" else "remainder"
    return next((family for family, matched in rules if matched), default)


def _linear(x, y, target: float) -> float:
    if target < x[0] or target > x[-1]:
        raise PortableModel1Error("interpolation would extrapolate")

    if target == x[0]:
        return float(y[0])

    for index in range(1, len(x)):
        if target <= x[index]:
            fraction = (target - x[index - 1]) / (x[index] - x[index - 1])
            return y[index - 1] + fraction * (y[index] - y[index - 1])

    return float(y[-1])


def _pchip_slopes(x, y) -> list[float]:
    """Fritsch-Butland slopes with the shape-preserving three-point endpoints."""
    h = [x[i + 1] - x[i] for i in range(len(x) - 1)]
    delta = [(y[i + 1] - y[i]) / h[i] for i in range(len(h))]
    if len(x) == 2:
        return [delta[0], delta[0]]
    slopes = [0.0] * len(x)
    for index in range(1, len(x) - 1):
        left, right = delta[index - 1], delta[index]
        if left == 0.0 or right == 0.0 or left * right <= 0.0:
            slopes[index] = 0.0
        else:
            w1 = 2.0 * h[index] + h[index - 1]
            w2 = h[index] + 2.0 * h[index - 1]
            slopes[index] = (w1 + w2) / (w1 / left + w2 / right)

    def endpoint(h0: float, h1: float, d0: float, d1: float) -> float:
        value = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if value * d0 <= 0.0:
            return 0.0
        if d0 * d1 < 0.0 and abs(value) > 3.0 * abs(d0):
            return 3.0 * d0
        return value

    slopes[0] = endpoint(h[0], h[1], delta[0], delta[1])
    slopes[-1] = endpoint(h[-1], h[-2], delta[-1], delta[-2])
    return slopes


def _pchip(x, y, target: float) -> float:
    if target < x[0] or target > x[-1]:
        raise PortableModel1Error("PCHIP would extrapolate")

    if target == x[-1]:
        return float(y[-1])
    upper = next(index for index, value in enumerate(x) if value > target)
    lower = upper - 1
    slopes = _pchip_slopes(x, y)
    width = x[upper] - x[lower]
    t = (target - x[lower]) / width
    h00 = (2.0 * t - 3.0) * t * t + 1.0
    h10 = ((t - 2.0) * t + 1.0) * t
    h01 = (-2.0 * t + 3.0) * t * t
    h11 = (t - 1.0) * t * t

    return (
        h00 * y[lower] + h10 * width * slopes[lower]
        + h01 * y[upper] + h11 * width * slopes[upper]
    )


def _pchip_derivative(x, y, target: float) -> float:
    """Analytic first derivative of the same C1 interpolant."""

    if target < x[0] or target > x[-1]:
        raise PortableModel1Error("PCHIP derivative would extrapolate")
    lower = (len(x) - 2 if target == x[-1] else
             next(index for index, value in enumerate(x) if value > target) - 1)
    upper = lower + 1
    width = x[upper] - x[lower]
    t = (target - x[lower]) / width
    slopes = _pchip_slopes(x, y)

    return ((6 * t * t - 6 * t) * y[lower] / width
            + (3 * t * t - 4 * t + 1) * slopes[lower]
            + (-6 * t * t + 6 * t) * y[upper] / width
            + (3 * t * t - 2 * t) * slopes[upper])


def _positive_matched_probabilities(mass, start, endpoint, left, left_slopes, right, right_slopes):
    """Quintic Bernstein value-and-slope match between two probability vectors.

    Interior controls lie on the segment between the two tangent controls.
    Every control must lie in the probability simplex (raise, never clip),
    so the matched vector is positive and closed.
    """
    length = endpoint - start
    c1 = {f: left[f] + length * left_slopes[f] / 5 for f in left}
    c4 = {f: right[f] - length * right_slopes[f] / 5 for f in left}
    controls = (dict(left), c1,
                {f: (2 * c1[f] + c4[f]) / 3 for f in left},
                {f: (c1[f] + 2 * c4[f]) / 3 for f in left},
                c4, dict(right))
    for row in controls:
        if (any(not math.isfinite(v) or v < -2e-12 or v > 1 + 2e-12 for v in row.values())
                or not math.isclose(math.fsum(row.values()), 1.0, abs_tol=2e-12, rel_tol=0)):
            raise PortableModel1Error("matching control is outside the probability simplex")
    x = (mass - start) / length
    weights = tuple(math.comb(5, i) * x**i * (1 - x)**(5 - i) for i in range(6))
    return {f: math.fsum(w * row[f] for w, row in zip(weights, controls)) for f in left}


class FamilyModel:
    """Portal contract, Pythia family response, source weights and saturation nodes (data/<model>/model1*.json)."""

    def __init__(self, data):
        self.__dict__.update(data)
        contract = self.contract = SimpleNamespace(**data["contract"])
        contract.families = tuple(SimpleNamespace(**item) for item in contract.families)
        contract.family_ids = tuple(family.family_id for family in contract.families)
        contract.named_family_ids = tuple(f for f in contract.family_ids if f != contract.remainder_family_id)
        contract.boundary_probabilities = {family.family_id: family.boundary_probability for family in contract.families}

    def _mixed_at(self, mass_gev: float, family_id: str) -> float:
        nodes, values = self.response_masses, self.mixed_probabilities[family_id]

        if min(values) > 0.0:
            return math.exp(_pchip(
                [math.log(node * node) for node in nodes],
                [math.log(value) for value in values],
                math.log(mass_gev * mass_gev),
            ))

        return _linear(nodes, values, mass_gev)

    def _matching_at(self, family_id: str, mass_gev: float) -> float:
        nodes = self.families[family_id]["matching_nodes"]

        if mass_gev <= nodes[0][0]:
            return nodes[0][1]

        return math.exp(_pchip(
            [math.log(mass * mass) for mass, _ in nodes],
            [math.log(value) for _, value in nodes],
            math.log(mass_gev * mass_gev),
        ))

    def _transport_at(self, mass: float, selected_power: float,
                      saturation_off: bool) -> tuple[dict[str, float], dict[str, float]]:
        scale = (self.contract.split_mass_gev / mass) ** (2 * selected_power)
        values, slopes = {}, {}

        for family in self.contract.named_family_ids:
            matching = 1.0 if saturation_off else self._matching_at(family, mass)
            log_slope = 0.0
            if not saturation_off:
                nodes = self.families[family]["matching_nodes"]
                if mass >= nodes[0][0]:
                    log_slope = _pchip_derivative(
                        [math.log(m * m) for m, _ in nodes],
                        [math.log(v) for _, v in nodes], math.log(mass * mass))
            values[family] = self.contract.boundary_probabilities[family] * scale * matching
            slopes[family] = values[family] * (2 * log_slope - 2 * selected_power) / mass
        remainder = self.contract.remainder_family_id
        values[remainder] = 1 - math.fsum(values.values())
        slopes[remainder] = -math.fsum(slopes.values())

        return values, slopes

    def evaluate(self, mass_gev: float, *, variation: str = "central") -> SimpleNamespace:
        """Family probabilities, source weights, raw proposal and event weights at one mass."""
        contract = self.contract
        mass = float(mass_gev)
        if not contract.split_mass_gev <= mass <= contract.endpoint_mass_gev:
            raise PortableModel1Error("evaluation lies outside the contract domain")
        if variation not in VARIATIONS:
            raise PortableModel1Error("variation must be central, saturation-off, p-low, or p-high")
        hook = contract.leading_power
        selected_power = float({"p-low": min(hook["alternatives"]),
                                "p-high": max(hook["alternatives"])}.get(variation, hook["central"]))
        saturation_off = variation == "saturation-off"
        remainder_id = contract.remainder_family_id
        matching_spec = contract.probability_matching
        kind = None if matching_spec is None else matching_spec["kind"]
        matched = None
        if kind == "coherent-native-kaon-independent-total":
            # B-L: the flat B-L rate model owns the active-pool probabilities.
            from ..b_l.rates import family_probabilities
            matched = family_probabilities(mass, self, selected_power, saturation_off)
        elif kind is not None and mass < matching_spec["endpoint_mass_gev"]:
            endpoint = matching_spec["endpoint_mass_gev"]
            right, right_slopes = self._transport_at(endpoint, selected_power, saturation_off)
            matched = _positive_matched_probabilities(
                mass, contract.split_mass_gev, endpoint, contract.boundary_probabilities,
                matching_spec["boundary_slopes_per_gev"], right, right_slopes)
        if matched is not None:
            named = {family: matched[family] for family in contract.named_family_ids}
            remainder = matched[remainder_id]
        else:
            scale = ((contract.split_mass_gev * contract.split_mass_gev) / (mass * mass)) ** selected_power
            boundary = contract.boundary_probabilities
            named = {family: boundary[family] * scale
                     * (1.0 if saturation_off else self._matching_at(family, mass))
                     for family in contract.named_family_ids}
            named_sum = math.fsum(named.values())
            if kind is not None:
                if named_sum > 1.0 + 2e-12:
                    raise PortableModel1Error("high-mass transported probabilities exceed unity")
                remainder = 1.0 - named_sum
            elif named_sum > 1.0:
                # Near an exhausted boundary vector, one common closure factor keeps
                # the relative named mixture; the remainder stays at zero.
                closure_factor = 1.0 / named_sum
                named = {family: value * closure_factor for family, value in named.items()}
                remainder = 0.0
            else:
                remainder = 1.0 - named_sum
        probabilities = {key: max(0.0, value) for key, value in {**named, remainder_id: remainder}.items()}
        probabilities[remainder_id] += 1.0 - math.fsum(probabilities.values())

        raw = {family: self._mixed_at(mass, family) for family in contract.family_ids}
        weights: dict[str, float] = {}
        for family in contract.family_ids:
            frozen = self.families.get(family)
            if frozen is not None and frozen["response_mode"] == BOUNDARY_CONDITIONED_RESPONSE:
                weights[family] = 0.0  # realized by a boundary generator, not by rejection
            elif raw[family] <= 0.0:
                if probabilities[family] > 2.0e-14:
                    raise PortableModel1Error(f"positive family {family} has zero generator support")
                weights[family] = 0.0
            else:
                weights[family] = probabilities[family] / raw[family]
        sources = {source: _linear(self.source_masses, self.source_weights[source], mass)
                   for source in contract.source_ids}
        return SimpleNamespace(mass_gev=mass, family_probabilities=probabilities, source_weights=sources,
                               raw_family_probabilities=raw, event_weights=weights)
