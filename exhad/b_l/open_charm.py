"""The light-hadron component of b-l plus a separately normalized open-charm contribution.

For L_int = g_BL V_mu J_BL^mu all widths below are divided by g_BL**2.
The charm-current spectral function is inferred from the measured EM charm
input (data/common/open_charm_card.json). Its unit-current width is multiplied by (Q_BL,c/Q_EM,c)**2
= (1/3 / (2/3))**2 = 1/4. No electromagnetic coupling e**2 belongs here.
This assumes charm-current dominance of open-charm production, neglecting
light-current production of charm and light/charm interference. It is not
an exact rotation of the inclusive electromagnetic current into B-L.

The light and charm pieces are distinguished BEFORE weak charm decays.
Consequently a charm decay is not vetoed merely because its light daughters
also occur in a prompt light-hadron channel. No electromagnetic final-state
ownership veto or light-isospin filter is applied to these weak daughters.
Subthreshold narrow charmonium poles are not added by this open-charm model.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
import random
from ..core.event_model import ConservedCharges, validate_complete_event
from ..core.event_worker import particle_catalog_from_pythia_xml
from ..core.eventcalc import six_field
from ..core.seeds import splitmix64
from ..dark_photon.charm_rates import build_open_charm_owner
from ..dark_photon.open_charm import PythiaCharmEventBackend, charm_primary_modes_at
from .light import BaryonLightGenerator, _mass

CHARM_CURRENT_WIDTH_SCALE = (1.0 / 3.0 / (2.0 / 3.0)) ** 2

_OUTER_TAG = 0xBB175B41

_LIGHT_TAG = 0xBB175B42

_CHARM_TAG = 0xBB175B43

_MODE_TAG = 0xBB175B44

_DECAY_TAG = 0xBB175B45


def _unit(seed):
    return (splitmix64(seed) >> 11) / float(1 << 53)


def _choose(items, weights, seed):
    target = _unit(seed) * math.fsum(weights)
    cumulative = 0.0
    for item, weight in zip(items, weights, strict=True):
        cumulative += weight
        if target < cumulative:
            return item
    raise RuntimeError("Cannot draw from an empty or nonpositive measure")


@dataclass(frozen=True)
class BaryonOpenCharmRatePoint:
    light: object
    charm_em: object | None

    @property
    def charm_width_per_g_b_squared_gev(self):
        return (0.0 if self.charm_em is None else
                CHARM_CURRENT_WIDTH_SCALE * self.charm_em.inclusive_charm_width)

    @property
    def inclusive_width_per_g_b_squared_gev(self):
        return (self.light.inclusive_width_per_g_b_squared_gev
                + self.charm_width_per_g_b_squared_gev)

    @property
    def charm_probability(self):
        return (self.charm_width_per_g_b_squared_gev
                / self.inclusive_width_per_g_b_squared_gev)


@dataclass(frozen=True)
class BaryonOpenCharmBatch:
    events: tuple
    origins: tuple[str, ...]
    rate_point: BaryonOpenCharmRatePoint


class BaryonOpenCharmGenerator:
    """Unweighted B-L hadronic decays, including open charm, at 2--5 GeV."""

    def __init__(self):
        self.light = BaryonLightGenerator()
        deployment = self.light.deployment
        self.charm = build_open_charm_owner()
        self.backend = PythiaCharmEventBackend(deployment.binary_path, deployment.xmldoc_path)
        self.catalog = particle_catalog_from_pythia_xml(deployment.xmldoc_path)

    def close(self):
        self.backend.close()
        self.light.close()

    def rate_point(self, mass_gev, *, variation="central"):
        mass = _mass(mass_gev)
        light = self.light.rate_point(mass, variation=variation)
        charm = (None if mass <= self.charm.open_charm_threshold_gev else
                 self.charm.closure_at(mass))
        return BaryonOpenCharmRatePoint(light, charm)

    def _charm_event(self, point, seed):
        # The measured families and unresolved residual exhaust the inclusive
        # charm width. Keep the selected family and charge mode on retries:
        # generator efficiency must not change their target probabilities.
        families = tuple(p for p in point.resolved if p.width > 0.0)
        selected = _choose((*families, None),
                           (*[p.width for p in families], point.residual_width), seed)
        mode = None

        if selected is not None:
            modes = charm_primary_modes_at(selected.channel_id, point.mass_gev)
            mode = _choose(modes, [p.relative_weight for p in modes], seed ^ _MODE_TAG)

        for attempt in range(1024):
            event_seed = splitmix64(seed ^ splitmix64(attempt))
            event = (self.backend.unresolved_event(point.mass_gev, event_seed)
                     if mode is None else self.backend.resolved_event(
                         point.mass_gev, mode.primary_pdgs, event_seed))
            if event is None:
                continue
            validate_complete_event(event, parent_mass_gev=point.mass_gev,
                                    target_charges=ConservedCharges(), catalog=self.catalog)
            return six_field(event, random.Random(splitmix64(seed ^ _DECAY_TAG)), fsum=True)
        raise RuntimeError("B-L charm generation exhausted attempts for the selected channel")

    def sample(self, mass_gev, count, *, seed=1, variation="central", active_pool=None):
        mass = _mass(mass_gev)

        if any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**64
               for value in (count, seed)):
            raise ValueError("count and seed must be unsigned 64-bit integers")
        rate = self.rate_point(mass, variation=variation)

        if rate.charm_probability == 0.0:
            # Where charm is closed, the request seed drives the light-hadron sample directly.
            light = self.light.sample(mass, count, seed=seed, variation=variation,
                                      active_pool=active_pool)
            return BaryonOpenCharmBatch(light.events, ("light-uds",) * count, rate)
        origins = tuple("open-charm" if _unit(seed ^ splitmix64(i) ^ _OUTER_TAG)
                        < rate.charm_probability else "light-uds" for i in range(count))
        light_count = origins.count("light-uds")
        light_events = iter(self.light.sample(
            mass, light_count, seed=splitmix64(seed ^ _LIGHT_TAG), variation=variation,
            active_pool=active_pool).events)
        events = tuple(next(light_events) if origin == "light-uds" else
                       self._charm_event(rate.charm_em, splitmix64(seed ^ splitmix64(i) ^ _CHARM_TAG))
                       for i, origin in enumerate(origins))

        return BaryonOpenCharmBatch(events, origins, rate)
