"""The light-quark (uds) component of the b-l hadronic decays on 2--5 GeV as EventCalc six-field records.

Each event draws one owner in proportion to the active fragmentation width
and the five external widths (rates.py).  Active events
come from the Model-1 sampler of the b-l deployment, whose
contract removes the external modes from the Pythia pool (exclusive-owner
veto).  External events are exclusive injections with the dark-photon
primary-mode tables, except that the K Kbar charged/neutral split uses the
native B-current amplitudes; the neutral pair of a vector is K_L K_S only.
"""

from __future__ import annotations
import functools
import math
import random
from typing import NamedTuple
from ..core.event_worker import EventRequest, ExclusiveInjectionRuntime
from ..core.eventcalc import six_field
from ..core.seeds import derive_event_seeds, splitmix64
from ..dark_photon.modes import DarkPhotonModeAuthority, ExclusiveMode
from ..model1.sampler import load_deployment, sample_deployment
from .rates import active_width, external_widths, kaon_pair_widths, total_width

ACTIVE_OWNER = "active-fragmentation"

# external block -> (realizer_id, request final-state support)
EXTERNAL_OWNERS = {
    "pi0-gamma-block": ("vector-pi0-gamma", ("22:1 111:1",)),
    "eta-gamma-block": ("vector-eta-gamma", ("22:1 221:1",)),
    "kaon-pair-block": ("vector-kaon-pair", ("-311:1 311:1", "-321:1 321:1", "130:1 310:1", "130:2", "310:2")),
    "proton-pair-block": ("vector-proton-pair", ("-2212:1 2212:1",)),
    "neutron-pair-block": ("vector-neutron-pair", ("-2112:1 2112:1",)),
}

_OWNER_DRAW_TAG = 0x42C0_5CC7_EEF0_6F31

_ACTIVE_BATCH_TAG = 0x6466_9A0C_39E4_20D7

_EXTERNAL_DECAY_TAG = 0xC53A_F76D_12A3_09B1


class BaryonLightError(ValueError):
    """A request for the light-hadron component of b-l was rejected or failed."""


class BaryonRatePoint(NamedTuple):
    inclusive_width_per_g_b_squared_gev: float
    owner_probabilities: dict[str, float]


class BaryonBatch(NamedTuple):
    events: tuple[tuple[float, ...], ...]
    rate_point: BaryonRatePoint


def _mass(value) -> float:
    mass = float(value)
    if not 2.0 <= mass <= 5.0:
        raise BaryonLightError(
            "B-L mass must lie in [2.00,5.00] GeV")
    return mass


def _closed(values: dict[str, float], target: float) -> dict[str, float]:
    """Nonnegative values whose fsum equals target exactly.

    The largest coordinate absorbs the residual (then single ulps).  If its
    correctly rounded sum jumps across target, a second positive coordinate is
    moved by at most 8 ulps (largest first) and the primary correction redone.
    """
    keys = tuple(values)
    original = [float(values[key]) for key in keys]
    index = max(range(len(original)), key=original.__getitem__)
    if (not math.isfinite(target) or target < 0.0
            or any(not math.isfinite(value) or value < 0.0 for value in original)
            or not target - math.fsum(v for i, v in enumerate(original) if i != index) >= 0.0):
        raise BaryonLightError("baryon-current width/probability closure failed")

    def settle(coordinates, attempts):
        corrected = target - math.fsum(v for i, v in enumerate(coordinates) if i != index)

        if corrected < 0.0 or not math.isfinite(corrected):
            return False
        coordinates[index] = corrected

        for _ in range(attempts):
            observed = math.fsum(coordinates)
            if observed == target:
                return True
            candidate = math.nextafter(coordinates[index], math.inf if observed < target else -math.inf)
            if candidate < 0.0 or candidate == coordinates[index]:
                return False
            coordinates[index] = candidate

        return False

    coordinates = list(original)
    if settle(coordinates, 64):
        return dict(zip(keys, coordinates))
    secondaries = sorted((i for i, value in enumerate(original) if i != index and value > 0.0),
                         key=original.__getitem__, reverse=True)
    for secondary_index in secondaries:
        for direction in (-math.inf, math.inf):
            secondary = original[secondary_index]
            for _ in range(8):
                secondary = math.nextafter(secondary, direction)
                if secondary < 0.0:
                    break
                coordinates = list(original)
                coordinates[secondary_index] = secondary
                if settle(coordinates, 4):
                    return dict(zip(keys, coordinates))
    raise BaryonLightError("failed exact baryon-current width/probability closure")


def _draw(probabilities: dict[str, float], seed: int) -> str:
    positive = [(key, value) for key, value in probabilities.items() if value > 0.0]

    if not positive:
        raise BaryonLightError("baryon-current owner measure is empty")
    target = (seed >> 11) / float(1 << 53)
    cumulative = 0.0

    for index, (key, probability) in enumerate(positive):
        cumulative = math.fsum((cumulative, probability))
        if target < cumulative or index + 1 == len(positive):
            return key


def _kaon_modes(mass: float) -> tuple[ExclusiveMode, ...]:
    charged, neutral = kaon_pair_widths(mass)
    total = math.fsum((charged, neutral))

    if total <= 0.0:
        raise BaryonLightError("B-current kaon pair has no positive mode")

    return tuple(
        ExclusiveMode(daughters, width / total, (key,), f"{charge}-kaon-pair")
        for charge, width, daughters, key in (
            ("charged", charged, (321, -321), "-321:1 321:1"),
            ("neutral", neutral, (130, 310), "130:1 310:1"))
        if width > 0.0)


def _mode_sources() -> dict:
    table = DarkPhotonModeAuthority()
    sources = {realizer: functools.partial(table.modes_for, realizer) for realizer, _ in EXTERNAL_OWNERS.values()}
    sources["vector-kaon-pair"] = _kaon_modes
    return sources


def _checked_events(records, mass: float) -> tuple[tuple[float, ...], ...]:
    """Finite six-field records closing to (0, 0, 0, mass) in the parent rest frame."""
    tolerance = 2.0e-8 * max(1.0, mass)
    events = []

    for record in records:
        values = tuple(float(value) for value in record)
        particles = [values[i:i + 6] for i in range(0, len(values), 6)]
        if (not values or len(values) % 6 or any(not math.isfinite(value) for value in values)
                or any(p[3] < 0.0 or p[4] < 0.0 or p[5] == 0.0 or not p[5].is_integer() for p in particles)
                or any(abs(math.fsum(p[k] for p in particles) - (mass if k == 3 else 0.0)) > tolerance
                       for k in range(4))):
            raise BaryonLightError("EventCalc event violates parent-rest-frame closure")
        events.append(values)

    return tuple(events)


class BaryonLightGenerator:
    """Unweighted light-uds decays of the b-l current at 2--5 GeV; close() stops its external workers."""

    def __init__(self):
        self.deployment = load_deployment("b-l")
        self.external = ExclusiveInjectionRuntime(
            self.deployment.binary_path, self.deployment.xmldoc_path, _mode_sources(),
            maximum_condition_attempts=128, maximum_workers=4)

    def close(self):
        self.external.close()

    def rate_point(self, mass_gev, *, variation="central") -> BaryonRatePoint:
        mass = _mass(mass_gev)
        if variation not in self.deployment.supported_variations:
            raise BaryonLightError("variation is not deployed")
        inclusive = total_width(mass)
        widths = _closed({ACTIVE_OWNER: active_width(mass), **external_widths(mass)}, inclusive)
        return BaryonRatePoint(inclusive, _closed(
            {key: value / inclusive for key, value in widths.items()}, 1.0))

    def sample(self, mass_gev, count, *, seed=1, variation="central", active_pool=None) -> BaryonBatch:
        rate = self.rate_point(mass_gev, variation=variation)
        mass = _mass(mass_gev)
        active, external = [], []

        for index in range(count):
            seeds = derive_event_seeds(seed, index)
            owner = _draw(rate.owner_probabilities, seeds.owner ^ _OWNER_DRAW_TAG)
            if owner == ACTIVE_OWNER:
                active.append(index)
            else:
                external.append((index, owner, seeds.provider))
        records = {}

        if active:
            events = sample_deployment(
                "b-l", mass, len(active), seed=splitmix64(seed ^ _ACTIVE_BATCH_TAG),
                variation=variation, active_pool=active_pool)
            records.update(zip(active, events, strict=True))

        for index, block, request_seed in external:
            realizer_id, keys = EXTERNAL_OWNERS[block]
            event = self.external.generate(EventRequest(mass, request_seed, None, realizer_id, keys))
            records[index] = six_field(
                event, random.Random(splitmix64(request_seed ^ _EXTERNAL_DECAY_TAG)), fsum=True)

        return BaryonBatch(_checked_events((records[index] for index in range(count)), mass), rate)
