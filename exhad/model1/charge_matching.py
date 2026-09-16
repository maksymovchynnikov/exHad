"""Charge-channel matching inside an unchanged hadronic family.

The proposal is the normalized PCHIP of Pythia's channel series. Below the
matching end the target is the same positive quintic value-and-slope match
used for family totals; above it, symmetry-fixed charge partitions of the
proposal persist. Families absent from the card keep weight one.
"""

from __future__ import annotations
from functools import lru_cache
import math
from .families import PortableModel1Error, _pchip, _pchip_derivative, _positive_matched_probabilities, canonical_channel_key


class ConditionalChannelMatching:
    """Deployment card: start, end, expand_pdgs, families, weight_bound."""

    def __init__(self, data):
        self.__dict__.update(data, expand_pdgs=frozenset(data["expand_pdgs"]))

    @staticmethod
    def _project(entry, values):
        result = dict(values)
        for partition in entry["partitions"]:
            total = math.fsum(values[k] for k in partition)
            for key, fraction in partition.items():
                result[key] = total * fraction
        return result

    @staticmethod
    def _proposal(entry, mass):
        """Normalized channel proposal and its mass derivative."""
        masses, series = entry["masses"], entry["series"]
        values = {k: _pchip(masses, values, mass) for k, values in series.items()}
        slopes = {k: _pchip_derivative(masses, values, mass) for k, values in series.items()}
        total, derivative = math.fsum(values.values()), math.fsum(slopes.values())
        return ({k: v / total for k, v in values.items()},
                {k: (slopes[k] * total - v * derivative) / total**2 for k, v in values.items()})

    @lru_cache(maxsize=512)
    def at(self, mass):
        """Per-family channel weights target/proposal and their maximum (at least one)."""

        if not math.isfinite(mass) or mass < self.start:
            raise PortableModel1Error("conditional matching queried below its support")
        weights = {}

        for family, entry in self.families.items():
            if mass >= self.end and not entry["partitions"]:
                continue
            proposal, _ = self._proposal(entry, mass)
            target = (self._project(entry, proposal) if mass >= self.end else
                      _positive_matched_probabilities(
                          mass, self.start, self.end, entry["left"], entry["slopes"],
                          entry["right"], entry["right_slopes"]))
            weights[family] = {key: target[key] / proposal[key] for key in entry["keys"]}
        maximum = max([1.] + [value for row in weights.values() for value in row.values()])

        if maximum > self.weight_bound * (1 + 2e-12):
            raise PortableModel1Error("conditional event weight exceeds its certified bound")

        return weights, maximum

    def weight(self, mass, family, pdgs):
        weights, _ = self.at(mass)
        if family not in weights:
            return 1.
        key = canonical_channel_key(pdgs)
        if key not in weights[family]:
            raise PortableModel1Error(f"unresolved channel in matched family {family}: {key}")
        return weights[family][key]
