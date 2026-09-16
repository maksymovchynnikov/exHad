"""Complete EventCalc events from a Model-1 deployment.

Fixed parent: draw one outer-owner authority row per event. External and
charm rows go to their realizers. An active event first draws a
boundary-conditioned family with probability sum P_cond; otherwise it is
rejection-sampled from source-mixed Pythia proposals with weight w_F c_F,key.
HNL exact W: the same active-pool sampler at each event's W, with the
conditioned families realized by the HNL boundary generator.
"""

from __future__ import annotations
from contextlib import ExitStack
import functools
import hashlib
import importlib
import json
import math
from pathlib import Path
import struct
from types import SimpleNamespace

from .. import DATA, ROOT
from ..core.event_worker import EventRequest, PythiaComponentRoute, PythiaSubprocessRuntime
from ..core.eventcalc import decay_rng, six_field, spacelike
from ..core.ownership import derive_ownership_cut
from ..core.seeds import MASK64, splitmix64
from ..models import SourceQuantumNumbers
from .charge_matching import ConditionalChannelMatching
from .families import BOUNDARY_CONDITIONED_RESPONSE, FamilyModel, PortableModel1Error, _linear, canonical_channel_key, classify_channel
from .realizers import RUNTIMES

_UINT64_DENOMINATOR = float(1 << 64)
# Seed-stream tags (fixed: they define the random streams).
_OWNER_TAG = 0x9E3779B97F4A7C15
_SOURCE_TAG = 0xD1B54A32D192ED03
_PROPOSAL_TAG = 0x94D049BB133111EB
_ACCEPT_TAG = 0xBF58476D1CE4E5B9
_REALIZER_TAG = 0x632BE59BD9B4E019
_CONDITIONED_OWNER_TAG = 0xA0761D6478BD642F
_CONDITIONED_FAMILY_TAG = 0xE7037ED1A0B428DB
_HNL_OWNER_TAG = 0xDB4F0B9175AE2165
_HNL_FAMILY_TAG = 0xBBE0563303A4615F

HNL_DEPLOYMENTS = {"CC_ud": "hnl-cc-ud", "NC_ud": "hnl-nc-ud"}

# Conditioned families the HNL boundary generator can realize at exact W.
HNL_CONDITIONED_FAMILIES = {"CC_ud": frozenset({"two-pion", "kaon-pair"}), "NC_ud": frozenset({"two-pion"})}


class PortableModel1DeploymentError(PortableModel1Error):
    """A deployment or complete-event route failed closed."""


@functools.lru_cache(maxsize=None)
def load_deployment(name: str) -> SimpleNamespace:
    """A deployment's data file and family model, bound to this tree's native build."""
    spec = json.loads((DATA / "common/model1_deployments.json").read_text())["deployments"][name]
    data = json.loads((DATA / spec["path"]).read_text())
    rows, model, matching = data.pop("rows"), data.pop("family_model"), data.pop("conditional_matching", None)

    return SimpleNamespace(
        **data, seed_namespace=spec["seed_namespace"],
        rows={row_id: SimpleNamespace(**row) for row_id, row in rows.items()},
        family_model=FamilyModel(model),
        binary_path=ROOT / "cpp/exhad",
        xmldoc_path=Path(json.loads((ROOT / ".runtime/current.json").read_text())["xmldoc"]),
        conditional_matching=None if matching is None else ConditionalChannelMatching(matching))


def _required_rejection_attempts(acceptance: float) -> int:
    """Smallest N with (1-A)^N <= 2^-64, rounded up in log space."""

    if acceptance >= 1.0:
        return 1
    log_tail, log_limit = math.log1p(-acceptance), -64 * math.log(2.0)
    attempts = max(1, math.ceil(log_limit / log_tail))

    while attempts * log_tail > log_limit:
        attempts += 1

    return attempts


def _point_rejection_attempt_budget(point, excluded_families=frozenset()):
    """Envelope max(w_F) and attempt budget of the normalized proposal q/Q.

    Excluded (boundary-conditioned) families have no Pythia proposal. The
    acceptance A = sum_E q_F w_F / (Q max_E w_F) is rounded down.
    """
    raw = point.raw_family_probabilities
    eligible = [family for family in raw if family not in excluded_families]
    target = {family: max(0.0, point.family_probabilities[family]) for family in eligible}
    weights = {family: max(0.0, point.event_weights[family]) for family in eligible}
    if any(raw[family] == 0.0 and target[family] > 0.0 for family in eligible):
        raise PortableModel1DeploymentError("positive target families lack generator support")
    raw_total = math.fsum(raw.values())
    target_total = math.fsum(target.values())
    envelope = max(weights.values())
    weighted_target = math.fsum(raw[family] * weights[family] for family in eligible)
    if min(raw_total, target_total, envelope, weighted_target) <= 0.0:
        raise PortableModel1DeploymentError("active rejection sampler has zero acceptance")
    if not math.isclose(weighted_target, target_total, rel_tol=2.0e-12, abs_tol=2.0e-14):
        # Guards against conditioned families leaking into the Pythia pool.
        raise PortableModel1DeploymentError("frozen event weights do not reproduce the eligible target mass")
    acceptance = math.nextafter(
        math.nextafter(weighted_target, 0.0) / math.nextafter(raw_total * envelope, math.inf), 0.0)
    attempts = _required_rejection_attempts(acceptance)
    if attempts > 1 << 20:
        raise PortableModel1DeploymentError(f"rejection-attempt budget {attempts} exceeds the hard cap")
    return {"maximum_attempts": attempts, "acceptance_probability_lower_bound": acceptance,
            "envelope": envelope}


def _joint_rejection_envelope(point, excluded_families, conditional_weights, base_envelope):
    """Bound the family-times-charge weights; exactly base_envelope without charge matching."""

    if not conditional_weights:
        return float(base_envelope)
    envelope = max((weight * factor for family, weight in point.event_weights.items()
                    if family not in excluded_families
                    for factor in conditional_weights.get(family, {"unmatched": 1.0}).values()), default=0.0)

    if envelope <= 0.0:
        raise PortableModel1DeploymentError("zero joint rejection envelope")
    # C++ multiplies in binary64 too; round outward so acceptance stays <= 1.

    return math.nextafter(envelope, math.inf)


@functools.lru_cache(maxsize=16)
def _read_outer_authority(path: Path):
    """Mass nodes, per-row probability columns (JSON floats) and the interpolation mode."""
    document = json.loads(path.read_text(encoding="utf-8"))
    return tuple(document["masses"]), document["probabilities"], document["interpolation"]


def _authority_probabilities(deployment, query: float) -> dict[str, float]:
    """Piecewise-linear outer-owner row probabilities; roundoff goes to the largest row."""
    masses, columns, interpolation = _read_outer_authority(DATA / deployment.outer_owner_authority)
    values = {row: _linear(masses, columns[row], query)
              for binding in deployment.rows.values() for row in binding.authority_row_ids}
    total = math.fsum(values.values())

    if total <= 0.0:
        raise PortableModel1DeploymentError("interpolated outer-owner measure is empty")

    if interpolation == "piecewise-linear-eventcalc-weights-normalized-at-query":
        values = {row: value / total for row, value in values.items()}
    values[max(values, key=values.get)] += 1.0 - math.fsum(values.values())

    return values


def _unit(value: int) -> float:
    return splitmix64(value & MASK64) / _UINT64_DENOMINATOR


def _draw(probabilities, word: int):
    draw, running = _unit(word), 0.0
    items = tuple(probabilities.items())
    for key, probability in items[:-1]:
        running += probability
        if draw < running:
            return key
    return items[-1][0]


def _master_seed(deployment, mass: float, count: int, seed: int, variation: str) -> int:
    payload = b"\0".join((deployment.seed_namespace.encode("ascii"),
                          struct.pack(">dQQ", mass, count, seed), variation.encode("ascii")))
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _parse_removed_key(key: str) -> tuple[int, ...]:
    """Sorted PDGs of a ``PDG:count`` channel key."""
    return tuple(sorted(int(pdg) for token in key.split()
                        for pdg, count in [token.split(":")] for _ in range(int(count))))


def _validate_six_field(events, expected: int) -> list[list[float]]:
    if len(events) != expected:
        raise PortableModel1DeploymentError("realizer returned wrong event count")
    return [[float(value) for value in event] for event in events]


def _external_realizer(deployment, row, authority_row_id: str, mass: float, seed: int) -> list[float]:
    module_name, function_name = row.realizer["callable"].split(":", 1)
    configuration = dict(row.realizer["configuration"],
                         runtime_binary=str(deployment.binary_path),
                         runtime_xmldoc=str(deployment.xmldoc_path))
    events = getattr(importlib.import_module(module_name), function_name)(
        mass_gev=mass, count=1, seed=seed, row_id=authority_row_id, configuration=configuration)
    return _validate_six_field(events, 1)[0]


def _route(component_id, runner):
    """One contract source; its card names the decaying particle's quantum numbers."""
    fields = dict(runner)
    return PythiaComponentRoute(
        component_id, SourceQuantumNumbers(**fields.pop("source_quantum_numbers")), **fields)


def _fragmentation_runtime(deployment, model):
    """Leased Pythia component runtime with one route per contract source."""
    routes = tuple(_route(source, deployment.source_runners[source])
                   for source in model.contract.source_ids)

    return RUNTIMES.use(
        ("fragmentation", deployment.seed_namespace, deployment.binary_path, deployment.xmldoc_path),
        lambda: PythiaSubprocessRuntime(
            deployment.binary_path, deployment.xmldoc_path, maximum_condition_attempts=64,
            maximum_workers=min(3, max(1, len(routes))), routes=routes,
            settings=deployment.generator_pythia_settings))


def _proposal_filters(deployment, point, excluded_families, envelope, charge_weights):
    """Per-source matching-filter lines of the compiled rejection sampler."""
    contract = deployment.family_model.contract
    common = [f"MODEL {contract.classifier_id} {contract.remainder_family_id}", f"ENVELOPE {envelope:.17g}"]

    for mode in contract.removed_external_modes:
        common += ["R " + family for family in mode["classifier_families"]]
        common += ["V " + canonical_channel_key(_parse_removed_key(key)) for key in mode["final_state_keys"]]

    for family, row in charge_weights.items():
        common += [f"C {family} {value:.17g} {key}" for key, value in row.items()]

    if charge_weights:
        common += [f"X {pdg}" for pdg in sorted(deployment.conditional_matching.expand_pdgs)]

    return {source: common + [
        f"F {spec.family_id} {(point.event_weights[spec.family_id] if spec.family_id not in excluded_families and source in spec.supported_sources else 0.):.17g}"
        for spec in contract.families] for source in point.source_weights}


def _conditioned_families(model) -> frozenset[str]:
    return frozenset(family for family, item in model.families.items()
                     if item["response_mode"] == BOUNDARY_CONDITIONED_RESPONSE)


def _conditioned_family(model, point, conditioned, event_seed: int, owner_tag: int, family_tag: int):
    """The boundary-conditioned family drawn for one active event, or None for the Pythia pool."""
    probability = math.fsum(point.family_probabilities[family] for family in conditioned)

    if not 0.0 <= probability <= 1.0 + 2.0e-12:
        raise PortableModel1DeploymentError("conditioned active probability is invalid")

    if probability > 0.0 and _unit(event_seed ^ owner_tag) < probability:
        return _draw({family: point.family_probabilities[family] / probability
                      for family in model.contract.named_family_ids
                      if family in conditioned and point.family_probabilities[family] > 0.0},
                     event_seed ^ family_tag)

    return None


def _active_event(deployment, model, runtime, mass, event_seed, variation,
                  excluded_families=frozenset(), prepared_point=None,
                  proposal_only=False, weight_floor_fraction=0.0):
    """Rejection-sample one active-pool event.

    Attempt a draws a source and a proposal seed; it is accepted when
    u < w_F c_F,key / envelope for a supported, non-removed, non-excluded
    family. The first accepted logical attempt is returned; a spacelike
    terminal is redrawn. Weighted proposals return (event, family, max(w, t)).
    """
    contract = model.contract
    point = prepared_point if prepared_point is not None else model.evaluate(mass, variation=variation)
    budget = _point_rejection_attempt_budget(point, excluded_families)
    attempts = budget["maximum_attempts"]
    conditional = deployment.conditional_matching
    charge_weights, charge_envelope = conditional.at(mass) if conditional is not None else ({}, 1.0)
    if charge_weights:
        attempts = _required_rejection_attempts(budget["acceptance_probability_lower_bound"] / charge_envelope)
    if attempts > deployment.maximum_attempts:
        raise PortableModel1DeploymentError(
            f"deployment rejection-attempt budget is insufficient; required={attempts}, "
            f"maximum={deployment.maximum_attempts}, mass_gev={mass:.17g}, variation={variation}")
    envelope = _joint_rejection_envelope(point, excluded_families, charge_weights, budget["envelope"])
    families = {item.family_id: item for item in contract.families}
    modes = contract.removed_external_modes
    removed_families = {family for mode in modes for family in mode["classifier_families"]}
    removed_keys = {canonical_channel_key(_parse_removed_key(key)) for mode in modes for key in mode["final_state_keys"]}
    forbidden = frozenset(key for mode in modes for key in mode["final_state_keys"])

    def proposal(attempt):
        attempt_seed = splitmix64(event_seed ^ _PROPOSAL_TAG ^ splitmix64(attempt))
        source = _draw(point.source_weights, attempt_seed ^ _SOURCE_TAG)
        uniform = _unit(attempt_seed ^ _ACCEPT_TAG)
        request = EventRequest(mass, attempt_seed, source, None, (), forbidden, None, attempt)
        return source, request, uniform * weight_floor_fraction if proposal_only else uniform

    def accepted(event, pdgs, source, uniform):
        classified = classify_channel(contract.classifier_id, pdgs)
        family = classified if classified in families else contract.remainder_family_id

        if (classified in removed_families or canonical_channel_key(pdgs) in removed_keys
                or family in excluded_families or source not in families[family].supported_sources):
            return None
        factor = point.event_weights[family]

        if family in charge_weights:
            factor *= conditional.weight(
                mass, family, derive_ownership_cut(event, expand_pdgs=conditional.expand_pdgs).pdg_ids)

        if not (factor > 0.0 and uniform < factor / envelope):
            return None

        return (event, family, max(factor, envelope * weight_floor_fraction)) if proposal_only else event

    filters = {source: "\n".join(lines) for source, lines in
               _proposal_filters(deployment, point, excluded_families, envelope, charge_weights).items()}
    # Each logical proposal reseeds its worker, so grouping a window by
    # source changes transport order only. A weighted request with a
    # floor below one normally needs a single positive-weight proposal.
    window = 1 if proposal_only and weight_floor_fraction < 1. else 32
    for start in range(0, attempts, window):
        grouped = {}
        for attempt in range(start, min(start + window, attempts)):
            source, request, uniform = proposal(attempt)
            grouped.setdefault(source, []).append((request, uniform))
        selected = None
        for source, items in grouped.items():
            if selected is not None:
                items = [item for item in items if item[0].attempt < selected[0]]
            if not items:
                continue
            result = runtime.generate_matched_batch(
                [item[0] for item in items], filters[source], [item[1] for item in items])
            if result is not None:
                index, event = result
                selected = (items[index][0].attempt, event, source, items[index][1])
        if selected is not None:
            _, event, source, uniform = selected
            if isinstance(event, Exception):
                raise event
            # Independent Python acceptance check of the returned graph.
            result = accepted(event, derive_ownership_cut(event).pdg_ids, source, uniform)
            if result is None or spacelike(event):
                raise PortableModel1DeploymentError("compiled sampler disagrees with reference acceptance")
            return result
    raise PortableModel1DeploymentError("compiled rejection sampler exhausted its adaptive budget")


def sample_deployment(deployment_name: str, mass: float, n: int, seed: int = 1,
                                         variation: str = "central", *, weighted: bool = False,
                                         weight_floor_fraction: float = 0.0, active_pool=None):
    """Return complete six-field events, or explicit importance-sampling records.

    Weighted mode keeps positive-weight proposals of the same proposal law
    (ownership, source and momenta unchanged). A floor t = fraction x envelope
    accepts with probability min(1, w/t) and returns max(w, t), so the product
    is exactly w. Raw weights are self-normalized within the fragmentation group
    over the whole sample; external and boundary-conditioned owners keep unit
    weight. Fraction zero keeps every positive-weight proposal; fraction one is
    the unweighted selection.

    An unweighted ``active_pool(deployment, point, variation, master, seeds)``
    realizes the Pythia-pool events {index: event seed} after all other owners,
    returning their six-field records (the accelerator of the alp-fermion, scalar and B-L portals).
    """
    if not weighted and weight_floor_fraction != 0.:
        raise ValueError("weight_floor_fraction requires explicit weighted mode")
    deployment = load_deployment(deployment_name)
    if deployment.portal_id.startswith("hnl-"):
        raise PortableModel1DeploymentError("HNL deployments are exact-W-only; use sample_hnl_exact_w")
    query = float(mass)
    if not deployment.activation_support_gev[0] <= query <= deployment.activation_support_gev[1]:
        raise PortableModel1DeploymentError("mass lies outside deployment support")
    if variation not in deployment.supported_variations:
        raise PortableModel1DeploymentError("variation is not deployed")
    master = _master_seed(deployment, query, n, seed, variation)
    # Draw the exact authority row: several rows may share one external realizer.
    authority = _authority_probabilities(deployment, query)
    owners = {row_id: row for row in deployment.rows.values() for row_id in row.authority_row_ids}
    model = deployment.family_model
    conditioned = _conditioned_families(model)
    point = model.evaluate(query, variation=variation)
    runtime = None
    generated, records, raw_weights, family_labels, pooled = {}, {}, {}, {}, {}
    with ExitStack() as leases:
        for index in range(n):
            event_seed = splitmix64(master ^ splitmix64(index))
            row_id = _draw(authority, event_seed ^ _OWNER_TAG)
            row = owners[row_id]
            if row.owner_kind == "active":
                family = _conditioned_family(model, point, conditioned, event_seed,
                                             _CONDITIONED_OWNER_TAG, _CONDITIONED_FAMILY_TAG)
                if family is None and active_pool is not None:
                    pooled[index] = event_seed
                    continue
                if family is None:
                    if runtime is None:
                        runtime = leases.enter_context(_fragmentation_runtime(deployment, model))
                    sampled = _active_event(deployment, model, runtime, query, event_seed, variation,
                                            excluded_families=conditioned, prepared_point=point,
                                            proposal_only=weighted, weight_floor_fraction=weight_floor_fraction)
                    if weighted:
                        sampled, family_labels[index], raw_weights[index] = sampled
                    generated[index] = sampled
                    continue
                row_id = family
            records[index] = _external_realizer(deployment, row, row_id, query,
                                                splitmix64(event_seed ^ _REALIZER_TAG))
    if pooled:
        records.update(active_pool(deployment, point, variation, master, pooled))
    if generated:
        rng = decay_rng(master)  # one EventCalc stable-decay stream for the whole active batch
        records.update((index, six_field(event, rng)) for index, event in generated.items())
    result = _validate_six_field([records[index] for index in range(n)], n)
    if not weighted:
        return result
    return {
        "events": result,
        "raw_weights": [raw_weights.get(i, 1.0) for i in range(n)],
        "normalization_groups": ["fragmentation" if i in raw_weights else "external" for i in range(n)],
        "family_labels": [family_labels.get(i, "external") for i in range(n)],
        "weight_convention": "self-normalized-within-fragmentation; external-unit-weight",
        "weight_floor_fraction": weight_floor_fraction,
    }


def sample_hnl_exact_w(current_id: str, w_values, seed: int = 1,
                                       variation: str = "central") -> list[list[float]]:
    """Generate one conditional HNL current event at each supplied exact W."""
    deployment = load_deployment(HNL_DEPLOYMENTS[current_id])

    if variation not in deployment.supported_variations:
        raise PortableModel1DeploymentError("variation is not deployed")
    model = deployment.family_model
    conditioned = _conditioned_families(model)

    if not conditioned <= HNL_CONDITIONED_FAMILIES[current_id]:
        raise PortableModel1DeploymentError(f"{current_id} has no boundary realizer for {sorted(conditioned)}")
    from ..hnl import boundary_decays
    values = [float(value) for value in w_values]
    result, categories = {}, {}
    runtime = None

    with ExitStack() as leases:
        for index, query in enumerate(values):
            if not deployment.activation_support_gev[0] <= query <= deployment.activation_support_gev[1]:
                raise PortableModel1DeploymentError(f"{current_id} exact W lies outside portable support")
            user_event_seed = splitmix64(seed ^ splitmix64(index) ^ int.from_bytes(struct.pack(">d", query), "big"))
            conversion_seed = _master_seed(deployment, query, 1, user_event_seed, variation)
            event_seed = splitmix64(conversion_seed ^ splitmix64(0))
            point = model.evaluate(query, variation=variation)
            family = _conditioned_family(model, point, conditioned, event_seed, _HNL_OWNER_TAG, _HNL_FAMILY_TAG)
            if family is not None:
                categories[index] = (family, query, event_seed)
                continue
            if runtime is None:
                runtime = leases.enter_context(_fragmentation_runtime(deployment, model))
            event = _active_event(deployment, model, runtime, query, event_seed, variation,
                                  excluded_families=conditioned, prepared_point=point)
            result[index] = six_field(event, decay_rng(conversion_seed))

    if categories:
        first_seed = next(iter(categories.values()))[2]
        realized = boundary_decays(
            current_id, [item[0] for item in categories.values()], [item[1] for item in categories.values()],
            splitmix64(first_seed ^ _REALIZER_TAG), False, deployment.binary_path, deployment.xmldoc_path)
        result.update(zip(categories, _validate_six_field(realized, len(categories))))

    return _validate_six_field([result[index] for index in range(len(values))], len(values))
