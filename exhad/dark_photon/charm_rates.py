"""Measured-data owner of electromagnetic open charm on 3.7--5.0 GeV.

The inclusive charm response is R_c = max(0, R_total - R_uds), zero below
the lowest measured-channel threshold, with R_total interpolated linearly
through the BES R-scan rows selected by the card from the PDG 2020 R table.
Published exclusive Born cross sections (linear only inside their measured
support) resolve part of it; if they exceed the inclusive cross section by
more than 64 ulp they are scaled down together, and the nonnegative
remainder is unresolved charm.  Cross sections are converted with the
pointlike alpha(0) muon cross section and widths with Gamma_mumu(m).
"""

from __future__ import annotations
import bisect
import csv
import json
import math
import shlex
from typing import NamedTuple
from .. import DATA, ROOT
from .widths import electromagnetic_uds_r, massive_muon_unit_current_width

ALPHA_ZERO = 1.0 / 137.035999084

GEV_MINUS_TWO_TO_NB = 389379.3656


class CharmChannel(NamedTuple):
    channel_id: str
    threshold_gev: float
    masses_gev: tuple[float, ...]
    sigmas_nb: tuple[float, ...]


class CharmCard(NamedTuple):
    domain_gev: tuple[float, float]
    r_masses_gev: tuple[float, ...]
    total_r: tuple[float, ...]
    channels: tuple[CharmChannel, ...]


class CharmContribution(NamedTuple):
    channel_id: str
    sigma_nb: float
    width: float


class CharmOwnerPoint(NamedTuple):
    mass_gev: float
    inclusive_charm_width: float
    resolved: tuple[CharmContribution, ...]
    residual_width: float


def _linear(masses: tuple[float, ...], values: tuple[float, ...], mass: float) -> float:
    right = bisect.bisect_right(masses, mass)
    if right == len(masses):
        return values[-1]
    if mass == masses[right - 1]:
        return values[right - 1]
    fraction = (mass - masses[right - 1]) / (masses[right] - masses[right - 1])
    return values[right - 1] + fraction * (values[right] - values[right - 1])


class OpenCharmOwner:
    """Compose measured resolved charm with the nonnegative inclusive closure."""

    def __init__(self, card: CharmCard) -> None:
        self.card = card
        self.open_charm_threshold_gev = min(channel.threshold_gev for channel in card.channels)

    def closure_at(self, mass_gev: float) -> CharmOwnerPoint:
        mass = float(mass_gev)
        low, high = self.card.domain_gev

        if not low <= mass <= high:
            raise ValueError(f"charm mass {mass} lies outside [{low}, {high}] GeV")
        total_r = _linear(self.card.r_masses_gev, self.card.total_r, mass)
        charm_r = 0.0 if mass < self.open_charm_threshold_gev else max(0.0, total_r - electromagnetic_uds_r(mass))
        muon_sigma = 4.0 * math.pi * ALPHA_ZERO * ALPHA_ZERO / (3.0 * mass * mass) * GEV_MINUS_TWO_TO_NB
        muon_width = massive_muon_unit_current_width(mass)
        inclusive_sigma = charm_r * muon_sigma
        raw = [0.0 if mass < channel.threshold_gev or not channel.masses_gev[0] <= mass <= channel.masses_gev[-1]
               else _linear(channel.masses_gev, channel.sigmas_nb, mass) for channel in self.card.channels]
        raw_sigma = math.fsum(raw)
        tolerance = 64.0 * math.ulp(max(1.0, inclusive_sigma, raw_sigma))
        scale = inclusive_sigma / raw_sigma if raw_sigma > inclusive_sigma + tolerance else 1.0
        resolved = tuple(CharmContribution(channel.channel_id, sigma * scale, sigma * scale / muon_sigma * muon_width)
                         for channel, sigma in zip(self.card.channels, raw))
        resolved_sigma = math.fsum(item.sigma_nb for item in resolved)
        residual_sigma = inclusive_sigma - resolved_sigma

        if residual_sigma < -tolerance:
            raise RuntimeError("open-charm projection failed nonnegative closure")
        residual_sigma = max(residual_sigma, 0.0)
        residual_width = residual_sigma / muon_sigma * muon_width
        inclusive_width = charm_r * muon_width

        if not (math.isclose(math.fsum((resolved_sigma, residual_sigma)), inclusive_sigma, rel_tol=2.0e-15, abs_tol=2.0e-15)
                and math.isclose(math.fsum((resolved_sigma / muon_sigma * muon_width, residual_width)), inclusive_width,
                                 rel_tol=2.0e-15, abs_tol=2.0e-15)):
            raise RuntimeError("open-charm closure is not exact")

        return CharmOwnerPoint(mass, inclusive_width, resolved, residual_width)


def build_open_charm_owner() -> OpenCharmOwner:
    """Load the frozen card, its selected inclusive-R rows and the exclusive cross sections."""

    document = json.loads((DATA / "common/open_charm_card.json").read_text())
    selections = document["inclusive"]["selection"]
    r_nodes = []
    for line in (ROOT / document["inclusive"]["source_path"]).read_text().splitlines():
        if not line.strip() or line.strip().startswith("*"):
            continue
        fields = shlex.split(line)
        if len(fields) < 11:
            continue
        try:
            mass, total_r = float(fields[0]), float(fields[3])
        except ValueError:
            continue
        code = fields[7].strip()
        if any(selection["short_code"] == code
               and ("range_gev" in selection and selection["range_gev"][0] <= mass <= selection["range_gev"][1]
                    or any(math.isclose(mass, value, rel_tol=0.0, abs_tol=1.0e-9)
                           for value in selection.get("masses_gev", ())))
               for selection in selections):
            r_nodes.append((mass, total_r))
    r_nodes.sort(key=lambda node: node[0])
    if len({mass for mass, _ in r_nodes}) != len(r_nodes):
        raise ValueError("duplicate selected inclusive-R mass")
    grouped = {channel["channel_id"]: [] for channel in document["channels"]}
    with (ROOT / document["exclusive_data_path"]).open(newline="") as stream:
        for row in csv.DictReader(stream):
            grouped[row["channel_id"].strip()].append((float(row["mass_gev"]), float(row["sigma_nb"])))
    channels = []
    for channel in document["channels"]:
        nodes = sorted(grouped[channel["channel_id"]], key=lambda node: node[0])
        channels.append(CharmChannel(channel["channel_id"], float(channel["physical_threshold_gev"]),
                                     tuple(mass for mass, _ in nodes), tuple(sigma for _, sigma in nodes)))
    domain = document["domain_gev"]
    return OpenCharmOwner(CharmCard((float(domain[0]), float(domain[1])), tuple(mass for mass, _ in r_nodes),
                                    tuple(value for _, value in r_nodes), tuple(channels)))
