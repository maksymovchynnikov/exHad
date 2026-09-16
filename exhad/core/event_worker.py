"""Persistent C++ event-worker client for complete Pythia 8.317 decay forests.

An ``exhad <2J> <P> <C|none> <charge> <qq|gg> ...`` process is fixed to a source
component, a unit injection table or the charm continuum; no Pythia initialization depends
on the mass, which travels with every request.  It answers
``REQUEST serial seed mass`` (and ``FILTER`` + ``BATCH`` for compiled matched
rejection) with one event graph.  Every request reseeds all Pythia streams, so
a pooled worker carries no event state and serves every mass; pool keys
separate every command line.  Each runtime keeps its own accept rule and seed
recipe.
"""

from __future__ import annotations
from collections import OrderedDict, deque
from contextlib import contextmanager
import functools
import math
import os
from pathlib import Path
import re
import select
import subprocess
import tempfile
import time
from typing import NamedTuple
from ..models import PARENT_SPIN_WORD, VECTOR_SOURCE, SourceQuantumNumbers
from .charge_completion import PionTopology
from .event_model import AncestryNode, ConservedCharges, EventModelError, FourMomentum, GeneratedEvent, ParticleDefinition, StableParticle, direct_pion_topology
from .ownership import ownership_key
from .seeds import splitmix64

WORKER_SCHEMA = "exhad-persistent-event-worker"

TIMEOUT_SECONDS = 60.0  # bounds a whole response, not one read

_MAX_GRAPH_LINES = 200_000

MULTIHADRON_CONTINUUM_STOP_MASS_GEV = 0.0

# The C++ worker adds 54321 to the base seed for its channel decayer.
_MAX_RUNNER_BASE_SEED = 900_000_000 - 54_321

_MODE_SELECTION_TAG = 0x3F84D5B5B5470917

_SETTING_PREFIXES = ("StringFlav:", "StringZ:", "StringPT:")

_SETTING_KEYS = frozenset({"StringFragmentation:stopNewFlav", "StringFragmentation:stopSmear"})

_FORCED_STABLE_PDGS = frozenset({-2112, -321, -211, -13, 13, 111, 130, 211, 310, 321, 2112})

_PARTICLE_BLOCK = re.compile(r"<particle\s+([^>]*)>(.*?)</particle>", re.DOTALL)

_CHANNEL_TAG = re.compile(r"<channel\s+([^>]*)/>", re.DOTALL)

_XML_ATTRIBUTE = re.compile(r"([A-Za-z][A-Za-z0-9]*)=\"([^\"]*)\"")


class PythiaSubprocessError(ValueError):
    """Raised when a worker or runtime cannot return a valid conditioned event."""


class EventRequest(NamedTuple):
    """One provider call; ownership keys are channel-level, the event is fully decayed."""

    mass_gev: float
    seed: int
    component_id: str | None = None
    realizer_id: str | None = None
    allowed_final_state_keys: tuple[str, ...] = ()
    forbidden_final_state_keys: frozenset[str] = frozenset()
    pion_condition: PionTopology | None = None  # exact direct-primary pion topology
    attempt: int = 0


class PythiaComponentRoute(NamedTuple):
    """Binding of one source component to its C++ projector and stop mass."""

    component_id: str
    source_quantum_numbers: SourceQuantumNumbers
    carries_parent_spin: bool
    decay_mode: str
    runner_component: str
    stop_mass_gev: float | None

# Stop mass 0 on the light multihadron continuum; charm routes keep Pythia's default.
DARK_PHOTON_ROUTES = (
    PythiaComponentRoute("isovector-vector", VECTOR_SOURCE, True, "qq", "rho", MULTIHADRON_CONTINUUM_STOP_MASS_GEV),
    PythiaComponentRoute("isoscalar-light-vector", VECTOR_SOURCE, True, "qq", "omega", MULTIHADRON_CONTINUUM_STOP_MASS_GEV),
    PythiaComponentRoute("isoscalar-strange-vector", VECTOR_SOURCE, True, "qq", "phi", MULTIHADRON_CONTINUUM_STOP_MASS_GEV),
)


def worker_settings(settings) -> tuple[str, ...]:
    """Whitelisted fragmentation settings with last-write-wins, in order of last occurrence."""

    effective: dict[str, str] = {}
    for raw in settings:
        key, value = (item.strip() for item in raw.split("=", 1))
        if not value or not (key.startswith(_SETTING_PREFIXES) or key in _SETTING_KEYS):
            raise PythiaSubprocessError(
                "fragmentation settings must be StringFlav, StringZ, StringPT, "
                "or StringFragmentation stopNewFlav/stopSmear")
        effective.pop(key, None)
        effective[key] = value
    return tuple(f"{key}={value}" for key, value in effective.items())


def _charges(pdg_id: int, electric: int) -> ConservedCharges:
    absolute, sign = abs(pdg_id), (1 if pdg_id > 0 else -1)
    quarks = ((absolute // 1000) % 10, (absolute // 100) % 10, (absolute // 10) % 10)
    baryon = sign if absolute < 1_000_000_000 and all(1 <= value <= 6 for value in quarks) else 0
    return ConservedCharges(electric, baryon, *(
        sign if absolute in pair else 0 for pair in ((11, 12), (13, 14), (15, 16))))


def particle_catalog_from_pythia_xml(xmldoc) -> dict[int, ParticleDefinition]:
    """Integer-charge species of ParticleData.xml; stable if forced stable or without an active channel."""

    catalog: dict[int, ParticleDefinition] = {}
    document = (Path(xmldoc).resolve() / "ParticleData.xml").read_text(encoding="utf-8")
    for header, body in _PARTICLE_BLOCK.findall(document):
        attributes = dict(_XML_ATTRIBUTE.findall(header))
        pdg_id, charge_type = int(attributes["id"]), int(attributes["chargeType"])
        # Graphs start at hadrons: no fractional-charge partons or diquarks.
        if pdg_id == 0 or charge_type % 3:
            continue
        active = any(
            int(channel.get("onMode", "1")) != 0 and float(channel.get("bRatio", "0")) > 0.0
            for channel in (dict(_XML_ATTRIBUTE.findall(tag)) for tag in _CHANNEL_TAG.findall(body)))
        for sign in ((1, -1) if "antiName" in attributes else (1,)):
            code = sign * pdg_id
            if code in catalog:
                raise EventModelError("particle catalog PDG identifiers must be unique")
            catalog[code] = ParticleDefinition(
                float(attributes.get("m0", "0")), _charges(code, sign * charge_type // 3),
                code in _FORCED_STABLE_PDGS or not active)
    return catalog


def _parse_graph(lines: list[str]) -> GeneratedEvent:
    """Reduce one event graph to the hadronic decay forest of its marked primary rows."""

    rows = [line.split() for line in lines if line.startswith("N ")]
    if sum(line.startswith("E ") for line in lines) != 1:
        raise PythiaSubprocessError("worker graph must hold exactly one event")
    momenta = []
    for index, tokens in enumerate(rows):
        if len(tokens) != 15 or int(tokens[1]) != index or tokens[8] not in ("0", "1") or tokens[9] not in ("0", "1"):
            raise PythiaSubprocessError("malformed or noncontiguous Pythia graph node")
        momentum = FourMomentum(*map(float, tokens[10:14]))
        if not (all(map(math.isfinite, momentum)) and momentum.energy_gev > 0.0):
            raise EventModelError("graph momentum must be finite with strictly positive energy")
        momenta.append(momentum)
    roots = {index for index, tokens in enumerate(rows) if tokens[9] == "1"}
    if not roots:
        raise PythiaSubprocessError("Pythia graph has no marked hadron roots")
    included, parent = set(roots), {}
    for index, tokens in enumerate(rows):
        if index in roots:
            continue
        mothers = {value for value in (int(tokens[4]), int(tokens[5])) if value > 0} & included
        if len(mothers) > 1:
            raise PythiaSubprocessError("Pythia decay descendant has multiple hadronic parents")
        if mothers:
            included.add(index)
            parent[index] = mothers.pop()
    finals = [index for index, tokens in enumerate(rows) if tokens[8] == "1" and int(tokens[2]) != 0]
    if not finals or not included.issuperset(finals):
        raise PythiaSubprocessError("a stable Pythia particle is outside the marked hadronic forest")
    if set(finals) != included.difference(parent.values()):
        raise EventModelError("ancestry leaves and terminal particles must coincide")
    order = sorted(included)
    compact = {row: node for node, row in enumerate(order)}
    terminal = {row: position for position, row in enumerate(finals)}
    return GeneratedEvent(
        tuple(AncestryNode(compact[row], int(rows[row][2]), compact[parent[row]] if row in parent else None,
                           terminal.get(row)) for row in order),
        tuple(StableParticle(int(rows[row][2]), momenta[row]) for row in finals))


class WorkerProcess:
    """One exhad stdio process read line by line; a failure closes it."""

    def __init__(self, binary, xmldoc, *arguments):
        self.lines, self.partial = deque(), b""
        self.errors = tempfile.TemporaryFile(mode="w+t")
        self.process = subprocess.Popen(
            (str(Path(binary).resolve()), *arguments), env=dict(os.environ, PYTHIA8DATA=str(Path(xmldoc).resolve())),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.errors, text=True, bufsize=1)

    def line(self, deadline: float | None, phase: str) -> str:
        """Next response line; every line of a response shares one deadline (None: unbounded)."""

        while not self.lines:
            output = self.process.stdout.fileno()
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0 or not select.select([output], [], [], remaining)[0]:
                    self.close()
                    raise PythiaSubprocessError(f"event worker timed out during {phase}")
            chunk = os.read(output, 1 << 16)
            if not chunk:
                self.errors.seek(0)
                detail = self.errors.read()[-2000:]
                self.close()
                raise PythiaSubprocessError("event worker closed its response: " + detail)
            data = self.partial + chunk
            cut = data.rfind(b"\n") + 1
            self.lines.extend(data[:cut].decode().splitlines())
            self.partial = data[cut:]

        return self.lines.popleft()

    def send(self, text: str) -> None:
        self.process.stdin.write(text)
        self.process.stdin.flush()

    def close(self) -> None:
        process = self.process
        try:  # communicate closes both pipes and reaps the process; a second close only waits
            process.communicate("QUIT\n" if process.returncode is None else None, timeout=2.0)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            process.kill()
            process.communicate()
        self.errors.close()


class EventWorker:
    """One fixed-configuration stdio worker; its READY record is checked once."""

    def __init__(self, binary, xmldoc, component="none", source_quantum_numbers=VECTOR_SOURCE,
                 carries_parent_spin=True, decay_mode="qq", *,
                 inject=None, injected_mass_mode="pole", flavor=None, stop_mass_gev=None, settings=(),
                 veto=frozenset()):
        self.serial = 0
        self.configuration = None
        options = [] if stop_mass_gev is None else [f"--stopMass={float(stop_mass_gev):.17g}"]
        options += [f"--set={setting}" for setting in settings]

        if inject is not None:  # one unit-probability mode
            options += [f"--inject={','.join(map(str, inject))}", f"--injectMasses={injected_mass_mode}"]

        if flavor is not None:  # the unfiltered component-none continuum
            options.append(f"--flavor={flavor}")
        options += [f"--veto={key}" for key in sorted(veto)]
        self.stdio = WorkerProcess(binary, xmldoc, *source_quantum_numbers.arguments, decay_mode,
                                   f"--component={component}",
                                   f"--parentSpin={PARENT_SPIN_WORD[bool(carries_parent_spin)]}",
                                   "--set=HadronLevel:mStringMin=1.0", *options)
        tokens = self.stdio.line(time.monotonic() + TIMEOUT_SECONDS, "startup").split()

        try:
            ready = (len(tokens) == 5 and tokens[:2] == ["READY", WORKER_SCHEMA]
                     and abs(float(tokens[2]) - 8.317) <= 1e-12
                     and tokens[3] == component and abs(float(tokens[4]) - 1.0) <= 1e-12)
        except ValueError:
            ready = False

        if not ready:
            self.close()
            raise PythiaSubprocessError("event worker returned an invalid READY record")

    def _graph(self, deadline: float) -> GeneratedEvent:
        lines = []
        while (line := self.stdio.line(deadline, "event graph")) != "X":
            lines.append(line)
            if len(lines) == _MAX_GRAPH_LINES:
                self.close()
                raise PythiaSubprocessError("event worker graph is unbounded")
        return _parse_graph(lines)

    def _end(self, serial: int, deadline: float) -> None:
        if self.stdio.line(deadline, "terminator") != f"END {serial}":
            self.close()
            raise PythiaSubprocessError("event worker response framing is invalid")

    def request(self, mass_gev, seed: int) -> GeneratedEvent | None:
        """One event at a mass for a logical seed, or None when the worker exhausts its attempts."""

        mapped = seed % _MAX_RUNNER_BASE_SEED + 1
        serial, self.serial = self.serial, self.serial + 1
        self.stdio.send(f"REQUEST {serial} {mapped} {float(mass_gev):.17g}\n")
        deadline = time.monotonic() + TIMEOUT_SECONDS
        tokens = self.stdio.line(deadline, "response").split()
        if len(tokens) != 4 or tokens[:2] != ["RESULT", str(serial)] or tokens[3] != str(mapped) \
                or tokens[2] not in ("EVENT", "EXHAUSTED"):
            self.close()
            raise PythiaSubprocessError("event worker rejected or misbound a request")
        event = self._graph(deadline) if tokens[2] == "EVENT" else None
        self._end(serial, deadline)
        return event

    def batch(self, mass_gev, proposals, configuration: str, attempts: int):
        """Compiled matched rejection: (index, event) of the first accepted proposal,
        (index, error) for a deferred C++ error bound to that proposal, or None."""

        deadline = time.monotonic() + TIMEOUT_SECONDS
        if configuration != self.configuration:
            self.stdio.send(f"FILTER {len(configuration.splitlines())}\n{configuration}\n")
            if self.stdio.line(deadline, "filter") != "FILTER-READY":
                raise PythiaSubprocessError("compiled matching configuration was rejected")
            self.configuration = configuration
        serial, self.serial = self.serial, self.serial + 1
        self.stdio.send(f"BATCH {serial} {len(proposals)} {attempts} {float(mass_gev):.17g}\n"
                        + "".join(f"{seed} {uniform:.17g}\n" for seed, uniform in proposals))
        tokens = self.stdio.line(deadline, "matched batch").split()
        if len(tokens) != 4 or tokens[0] not in ("BATCHRESULT", "BATCHERROR") or tokens[1] != str(serial):
            raise PythiaSubprocessError("invalid matched batch response")
        index = int(tokens[2])
        if not -1 <= index < len(proposals) or int(tokens[3]) != (0 if index < 0 else proposals[index][0]):
            raise PythiaSubprocessError("matched batch seed/index binding differs")
        result = None
        if tokens[0] == "BATCHERROR":
            if index < 0:
                raise PythiaSubprocessError("unbound matched batch error")
            result = index, PythiaSubprocessError(self.stdio.line(deadline, "matched proposal error"))
        elif index >= 0:
            result = index, self._graph(deadline)
        self._end(serial, deadline)
        return result

    def close(self) -> None:
        self.stdio.close()


class WorkerPool:
    """LRU of closable workers or runtimes; a failure while one is in use discards it."""

    def __init__(self, maximum_size: int) -> None:
        self.items = OrderedDict()
        self.maximum_size = maximum_size

    @contextmanager
    def use(self, key, create):
        item = self.items.get(key)

        if item is None:
            item = self.items[key] = create()
            while len(self.items) > self.maximum_size:
                self.items.popitem(last=False)[1].close()
        else:
            self.items.move_to_end(key)

        try:
            yield item
        except BaseException:
            if self.items.get(key) is item:
                del self.items[key]
            item.close()
            raise

    def request(self, key, create, mass_gev, seed: int) -> GeneratedEvent | None:
        with self.use(key, create) as worker:
            return worker.request(mass_gev, seed)

    def close(self) -> None:
        while self.items:
            self.items.popitem()[1].close()


def attempt_seeds(seed: int, attempts: int):
    """The request seed, then splitmix64(seed ^ splitmix64(k)) for k = 1 .. attempts-1."""

    return (seed if attempt == 0 else splitmix64(seed ^ splitmix64(attempt)) for attempt in range(attempts))


def retry(seeds, draw, accept) -> GeneratedEvent | None:
    """First drawn event that ``accept`` admits over the logical seeds; a None draw is a rejected attempt."""

    for seed in seeds:
        event = draw(seed)
        if event is not None and accept(event):
            return event
    return None


class PythiaSubprocessRuntime:
    """Projected source-component continuum.

    An ownership cut outside the convention is unrepresented ``other`` (key
    None); a forbidden key or a pion-condition mismatch retries.  After the
    budget, a pion condition gets one flat N-body phase-space event of its
    exact topology.  The C++ veto is always passed and re-checked here.
    """

    def __init__(self, binary, xmldoc, *, maximum_condition_attempts, maximum_workers,
                 routes=DARK_PHOTON_ROUTES, settings=()):
        self.pool = WorkerPool(maximum_workers)
        self.binary, self.xmldoc, self.attempts = binary, xmldoc, maximum_condition_attempts
        self.routes = {route.component_id: route for route in routes}
        if len(self.routes) != len(routes) or len({route.runner_component for route in routes}) != len(routes):
            raise PythiaSubprocessError("runtime component routes must be one-to-one")
        self.settings = worker_settings(settings)

    def close(self) -> None:
        self.pool.close()

    def _worker(self, component_id, veto):
        route = self.routes[component_id]
        key = (*route[1:], self.settings, tuple(sorted(veto)))
        return key, lambda: EventWorker(
            self.binary, self.xmldoc, route.runner_component, route.source_quantum_numbers,
            route.carries_parent_spin, route.decay_mode,
            stop_mass_gev=route.stop_mass_gev, settings=self.settings, veto=veto)

    def generate(self, request: EventRequest) -> GeneratedEvent:
        forbidden, condition = request.forbidden_final_state_keys, request.pion_condition

        def accept(event):
            return ownership_key(event) not in forbidden and (
                condition is None or direct_pion_topology(event) == condition)

        key, create = self._worker(request.component_id, forbidden)
        event = retry(attempt_seeds(request.seed, self.attempts),
                      functools.partial(self.pool.request, key, create, request.mass_gev), accept)
        if event is None and condition is not None:
            pdgs = (211,) * condition.n_plus + (111,) * condition.n_zero + (-211,) * condition.n_minus
            event = self.pool.request(
                ("pion-fallback", pdgs), lambda: EventWorker(self.binary, self.xmldoc, inject=pdgs),
                request.mass_gev, splitmix64(request.seed ^ splitmix64(self.attempts)))
            if event is not None and not accept(event):
                event = None
        if event is None:
            raise PythiaSubprocessError(
                f"Pythia reference provider exhausted its conditioning budget; mass_gev={request.mass_gev:.17g}, "
                f"component={request.component_id}, pion_condition={condition}, attempts={self.attempts}")
        return event

    def generate_matched_batch(self, requests, configuration: str, uniforms):
        """Compiled rejection over proposals sharing mass, source and vetoes; see EventWorker.batch."""

        first = requests[0]
        key, create = self._worker(first.component_id, first.forbidden_final_state_keys)
        with self.pool.use(key, create) as worker:
            result = worker.batch(first.mass_gev, [(request.seed, uniform) for request, uniform
                                                   in zip(requests, uniforms)], configuration, self.attempts)
        if result is not None and isinstance(result[1], Exception):
            self.pool.items.pop(key).close()
        return result


def _select_exclusive_mode(modes, request_seed: int):
    """Draw one supplied conditional mode without touching retry streams."""

    target = (splitmix64(request_seed ^ _MODE_SELECTION_TAG) >> 11) / float(1 << 53) \
        * math.fsum(mode.relative_weight for mode in modes)
    cumulative = 0.0
    for mode in modes:
        cumulative += mode.relative_weight
        if target < cumulative:
            return mode
    return modes[-1]


class ExclusiveInjectionRuntime:
    """Injection realizers of supplied exclusive modes.

    One mode is drawn per request before any retry; only its decays are
    retried until the ownership key lies in its target support (a cut outside
    the convention retries).  ``mode_sources`` maps realizer ids to mass ->
    modes callables; workers are shared by daughters.
    """

    def __init__(self, binary, xmldoc, mode_sources, *, maximum_condition_attempts, maximum_workers,
                 injected_mass_mode="pole"):
        self.pool = WorkerPool(maximum_workers)
        self.binary, self.xmldoc, self.attempts = binary, xmldoc, maximum_condition_attempts
        self.injected_mass_mode = injected_mass_mode
        self.modes_at = functools.lru_cache(maxsize=256)(
            lambda realizer_id, mass_gev: mode_sources[realizer_id](mass_gev))

    def close(self) -> None:
        self.pool.close()

    def generate(self, request: EventRequest, allowed_mode_ids: frozenset[str] | None = None) -> GeneratedEvent:
        modes = self.modes_at(request.realizer_id, request.mass_gev)

        if allowed_mode_ids is not None:
            # Mode ids can change with mass ('-combined'): never renormalize silently.
            missing = allowed_mode_ids.difference(mode.mode_id for mode in modes)
            if missing:
                raise PythiaSubprocessError(
                    f"exclusive mode subset is absent from the requested realizer: {', '.join(sorted(missing))}")
            modes = tuple(mode for mode in modes if mode.mode_id in allowed_mode_ids)
        mode = _select_exclusive_mode(modes, request.seed)
        support = frozenset(mode.target_ownership_keys)

        if not support <= frozenset(request.allowed_final_state_keys) or support & request.forbidden_final_state_keys:
            raise PythiaSubprocessError("selected exclusive mode target support escapes the request support")
        daughters = mode.daughter_pdgs
        event = retry(attempt_seeds(request.seed, self.attempts), functools.partial(
            self.pool.request, daughters, lambda: EventWorker(
                self.binary, self.xmldoc, inject=daughters, injected_mass_mode=self.injected_mass_mode),
            request.mass_gev), lambda event: ownership_key(event) in support)

        if event is None:
            raise PythiaSubprocessError("exclusive Pythia provider exhausted its ownership-conditioning budget")

        return event


class ExplicitPrimaryRuntime:
    """Pole-mass kinematics and decays of an already selected primary-hadron list."""

    def __init__(self, binary, xmldoc, *, maximum_condition_attempts, maximum_workers):
        self.pool = WorkerPool(maximum_workers)
        self.binary, self.xmldoc, self.attempts = binary, xmldoc, maximum_condition_attempts

    def close(self) -> None:
        self.pool.close()

    def generate(self, mass_gev, primary_pdgs, seed: int) -> GeneratedEvent:
        pdgs = tuple(sorted(primary_pdgs))

        def accept(event):
            if event.primary_pdgs != pdgs:
                raise PythiaSubprocessError("Pythia changed the selected primary-hadron identity")
            return True

        event = retry(attempt_seeds(seed, self.attempts), functools.partial(
            self.pool.request, pdgs, lambda: EventWorker(self.binary, self.xmldoc, inject=pdgs), mass_gev), accept)
        if event is None:
            raise PythiaSubprocessError("explicit-primary runtime exhausted its decay-conditioning budget")
        return event
