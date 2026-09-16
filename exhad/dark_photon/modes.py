"""Primary-mode mixtures for the dark-photon exclusive realizers.

A mode weight is R_table[label](min(m, 2 GeV)) * response_factor *
variant_factor, using the 29 native ReD-DeLiVeR R_mu^F tables; resolved
mixtures above 2 GeV are therefore frozen at the edge while the runtime
still generates at the request mass.  The persistent kaon-pair and K Kbar pi
realizers instead follow the native DeLiVeR form factors with the
electromagnetic current through 5 GeV: FK charged/neutral, and FKKpi modes
0-2 plus FPhiPi.  The FKKpi tables end at 4 GeV, so above it both are
continued by Gamma4 (4/m)^3 exp[(3 + 4 Gamma4'/Gamma4)(1 - 4/m)], matched in
value and in a 3-point backward slope.  Contributions with identical
(daughters, targets) are combined.  A runtime must select one mode once and
keep it across decay retries; redrawing would fold the response twice.
"""

from __future__ import annotations
import bisect
import functools
import importlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import NamedTuple
from .. import DATA, ROOT

FRAGMENTATION_EDGE_GEV = 2.00

KKPI_NATIVE_END_GEV = 4.0

KKPI_NATIVE_STEP_GEV = 0.005

KKPI_PARTIAL_WIDTH_ASYMPTOTIC_POWER = 3.0


class DarkPhotonModeError(ValueError):
    """Installed submode tables cannot define a physical mode mixture."""


class ExclusiveMode(NamedTuple):
    """One physical primary, its weight, and its exact ownership support."""

    daughter_pdgs: tuple[int, ...]
    relative_weight: float
    target_ownership_keys: tuple[str, ...]
    mode_id: str


@functools.lru_cache(maxsize=1)
def _primary_modes():
    """Mass nodes, the 29 native R tables by label and each realizer's mode specs (primary_modes.json)."""
    data = json.loads((DATA / "dark-photon" / "primary_modes.json").read_text())
    return data["masses"], data["tables"], {
        realizer_id: tuple(SimpleNamespace(**dict(spec, daughter_pdgs=tuple(spec["daughter_pdgs"]),
                                                  target_ownership_keys=tuple(spec["target_ownership_keys"])))
                           for spec in specs) for realizer_id, specs in data["realizers"].items()}


def _table_at(masses, values, mass_gev: float) -> float:
    """Linear interpolation of one R table between its nodes."""
    index = bisect.bisect_left(masses, mass_gev)

    if masses[index] == mass_gev:
        return values[index]
    left_mass, right_mass = masses[index - 1:index + 1]
    left_value, right_value = values[index - 1:index + 1]
    fraction = (mass_gev - left_mass) / (right_mass - left_mass)

    return left_value + fraction * (right_value - left_value)


def _native_form_factors(root: Path) -> list[object]:
    """Import DeLiVeR FK, FKKpi and FPhiPi from external/deliver and set the EM current once."""

    deliver = (root / "external" / "deliver").resolve()
    if str(deliver) not in sys.path:
        sys.path.insert(0, str(deliver))
    modules = []
    for name in ("FK", "FKKpi", "FPhiPi"):
        module = importlib.import_module("src.form_factors." + name)
        actual = Path(str(getattr(module, "__file__", ""))).resolve()
        if actual != (deliver / "src" / "form_factors" / (name + ".py")).resolve():
            raise DarkPhotonModeError(f"DeLiVeR module {name} resolved to {actual}")
        module.resetParameters(1.0, 0.0, 0.0, 0.0, 2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0)
        modules.append(module)
    return modules


class DarkPhotonModeAuthority:
    """Conditional primary-mode mixtures of the dark-photon realizers on [1.20, 5.00] GeV."""

    def __init__(self) -> None:
        self._masses, self._tables, self._specs = _primary_modes()
        self._kaon, self._kkpi, self._phi_pion = _native_form_factors(ROOT)
        self._kkpi_endpoints = {}

        for label in ("KKpi_0", "KKpi_1", "KKpi_2", "PhiPi"):
            at_endpoint, one_step_left, two_steps_left = (
                self._native_kkpi(label, KKPI_NATIVE_END_GEV - step * KKPI_NATIVE_STEP_GEV) for step in (0.0, 1.0, 2.0))
            slope = (3.0 * at_endpoint - 4.0 * one_step_left + two_steps_left) / (2.0 * KKPI_NATIVE_STEP_GEV)
            self._kkpi_endpoints[label] = at_endpoint, slope

    @property
    def realizer_ids(self) -> frozenset[str]:
        return frozenset(self._specs)

    def _native_kkpi(self, label: str, mass: float) -> float:
        if label == "PhiPi":
            return float(self._phi_pion.GammaDM(mass))
        return float(self._kkpi.GammaDM_mode(mass, int(label[-1])))

    def _kkpi_width(self, label: str, mass: float) -> float:
        if mass <= KKPI_NATIVE_END_GEV:
            return self._native_kkpi(label, mass)
        endpoint, endpoint_slope = self._kkpi_endpoints[label]
        correction = KKPI_PARTIAL_WIDTH_ASYMPTOTIC_POWER + KKPI_NATIVE_END_GEV * (endpoint_slope / endpoint)
        return (endpoint * (KKPI_NATIVE_END_GEV / mass) ** KKPI_PARTIAL_WIDTH_ASYMPTOTIC_POWER
                * math.exp(correction * (1.0 - KKPI_NATIVE_END_GEV / mass)))

    def modes_for(self, realizer_id: str, request_mass_gev: float) -> tuple[ExclusiveMode, ...]:
        """Return a positive normalized primary mixture for one realizer."""

        mass = float(request_mass_gev)
        if not 1.20 <= mass <= 5.00:
            raise DarkPhotonModeError("request mass must lie in [1.20,5.00] GeV")
        try:
            specs = self._specs[realizer_id]
        except KeyError as exc:
            raise DarkPhotonModeError(f"unknown dark-photon realizer {realizer_id!r}") from exc
        lookup = min(mass, FRAGMENTATION_EDGE_GEV)
        if realizer_id == "vector-kaon-pair" and mass > FRAGMENTATION_EDGE_GEV:
            values = {"KK_c": float(self._kaon.GammaDM_mode(mass, 1)), "KK_n": float(self._kaon.GammaDM_mode(mass, 0))}
        elif realizer_id == "vector-kaon-pair-pion" and mass > FRAGMENTATION_EDGE_GEV:
            values = {label: self._kkpi_width(label, mass) for label in self._kkpi_endpoints}
        else:
            values = {spec.source_label: _table_at(self._masses, self._tables[spec.source_label], lookup)
                      for spec in specs}
        raw = tuple((spec, values[spec.source_label] * spec.response_factor * spec.variant_factor) for spec in specs)
        total = math.fsum(weight for _, weight in raw)
        if total <= 0.0:
            # A one-mode channel can be closed at the lookup mass (NN below
            # threshold); its zero-width owner is never selected, but the
            # unique mode keeps preflight total.
            if len(specs) != 1:
                raise DarkPhotonModeError(f"realizer {realizer_id} has no positive mode at {lookup}")
            spec = specs[0]
            return (ExclusiveMode(spec.daughter_pdgs, 1.0, spec.target_ownership_keys, spec.mode_id),)
        # Combine only identical physical response blocks: the same daughters
        # can feed different ownership targets, which must stay separate.
        grouped: dict[tuple[tuple[int, ...], tuple[str, ...]], list[tuple[SimpleNamespace, float]]] = {}
        for spec, weight in raw:
            if weight > 0.0:
                grouped.setdefault((spec.daughter_pdgs, spec.target_ownership_keys), []).append((spec, weight))
        modes = []
        for (daughters, targets), contributions in grouped.items():
            mode_id = contributions[0][0].mode_id + ("" if len(contributions) == 1 else "-combined")
            modes.append(ExclusiveMode(daughters, math.fsum(weight for _, weight in contributions) / total,
                                       targets, mode_id))
        if not math.isclose(math.fsum(mode.relative_weight for mode in modes), 1.0,
                            rel_tol=0.0, abs_tol=8.0 * math.ulp(1.0)):
            raise RuntimeError("dark-photon primary mixture failed closure")
        return tuple(modes)
