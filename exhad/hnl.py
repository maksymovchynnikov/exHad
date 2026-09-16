"""HNL hadronization at the exact invariant mass W of the sampled q-qbar pair.

The host keeps the HNL rates and the lepton or neutrino four-vector; only the
q-qbar subsystem is replaced, event by event at its own W.  Routing by W:

* CC_ud: dedicated tables to 1.65 GeV, portable Model 1 on [1.651, 5] GeV; in the
  gap a cubic window (endpoint 2.0 GeV) mixes the 1.65-GeV categories with the
  projected string.
* NC_ud: dedicated tables to 3.0 GeV, portable Model 1 on [3.0, 5] GeV.
* NC_s: dedicated tables to 3.0 GeV; on 3.0-3.75 GeV a cubic window mixes the
  3.0-GeV composition (dilated to W) with the projected string; projected above.
* CC_us: tau-normalized resonance injection at every W.
* CC_cd, CC_cs: projected single-open-charm strings (the C++ worker vetoes the
  single D+ / Ds+ owned by EventCalc).

On 4-5 GeV a smootherstep hands events to showered Pythia, which takes every
W >= 5 GeV; the heavy currents NC_c, NC_b, CC_ub and CC_cb always use it.
"""
import atexit
import math
from functools import lru_cache
import numpy as np
from . import ROOT
from .core.pdg import antiparticle, charge, is_quark, m0
from .models import CHARGED_CURRENT_SOURCE, HNL_MASS_GEV, NEUTRAL_CURRENT_SOURCE

CC_PAIRS = {'CC_ud': (2, 1), 'CC_us': (2, 3), 'CC_cd': (4, 1), 'CC_cs': (4, 3), 'CC_ub': (2, 5), 'CC_cb': (4, 5)}
NC_CURRENTS = {1: 'NC_ud', 2: 'NC_ud', 3: 'NC_s', 4: 'NC_c', 5: 'NC_b'}
# The exhad router covers THRESHOLDS to 5.27 GeV (pi pi0, K pi0, D0 pi+, D0 K+, 2 pi, 2 K), which also
# anchor the charm worker bins.
THRESHOLDS = {'CC_ud': m0(211) + m0(111), 'CC_us': m0(321) + m0(111), 'CC_cd': m0(421) + m0(211),
              'CC_cs': m0(421) + m0(321), 'NC_ud': 2.0 * m0(211), 'NC_s': 2.0 * m0(321)}
# Sampled primaries must reach the router thresholds (NC_c: a sanity bound below PYTHIA_W_MIN);
PRIMARY_W_MIN = {**THRESHOLDS, 'NC_c': 2.0 * m0(411)}
# heavy currents go to showered Pythia above PYTHIA_W_MIN.
PYTHIA_W_MIN = {'NC_c': 3.75, 'NC_b': 10.56, 'CC_ub': 5.4196, 'CC_cb': 7.155}
ENDPOINTS = {'CC_ud': 2.0, 'NC_s': 3.75}  # projected-string endpoints of the duality windows
# Projected strings: the current's quantum numbers, the component table, and per
# component (runner, id, table column).  Those quantum numbers are the conservation-law
# filter the C++ runner builds; no name of the decaying particle reaches it.
PROJECTED = {
    'CC_ud': (CHARGED_CURRENT_SOURCE, 'cc_component_weights.csv',
              (('ccvec', 'cc-ud-vector', 'wVec'), ('ccax', 'cc-ud-axial', 'wAx'))),
    'NC_ud': (NEUTRAL_CURRENT_SOURCE, 'nc_component_weights.csv',
              (('rho', 'nc-ud-rho', 'wrho'), ('omega', 'nc-ud-omega', 'womega'), ('ncax', 'nc-ud-axial', 'wa1'))),
    'NC_s': (NEUTRAL_CURRENT_SOURCE, 'nc_component_weights.csv',
             (('phi', 'nc-s-vector', 'wphi'), ('ncsax', 'nc-s-axial', 'wf1s'))),
    'CC_cd': (CHARGED_CURRENT_SOURCE, None, (('cccd', 'cc-cd-continuum', None),)),
    'CC_cs': (CHARGED_CURRENT_SOURCE, None, (('cccs', 'cc-cs-continuum', None),)),
}
# Streams of the final-pass exact-W regeneration seeds ('CCUD', ..., 'NCLT', 'NCST').
RETRY_STREAMS = {'CC_ud': 0x43435544, 'CC_us': 0x43435553, 'CC_cd': 0x43434344, 'CC_cs': 0x43434353,
                 'NC_ud': 0x4E434C54, 'NC_s': 0x4E435354}
_RUNTIMES = {}  # projected-string runtime per current, closed at exit
atexit.register(lambda: [runtime.close() for runtime in _RUNTIMES.values()])


class _Closed(RuntimeError):
    """The selected final state is kinematically closed at the requested W."""


def channel_spec(pdgs):
    """(current, conjugate) of a signed N -> lepton + q qbar row, or None.

    CC output uses hadronic charge -1; conjugate is set by the hadronic electric charge.
    """
    ids = [int(p) for p in pdgs if int(p) != -999]
    quarks = [p for p in ids if is_quark(p)]
    neutrinos = [p for p in ids if abs(p) in (12, 14, 16)]
    charged = [p for p in ids if abs(p) in (11, 13, 15)]
    if len(ids) != 3 or len(quarks) != 2:
        return None
    if len(neutrinos) == 1 and not charged:
        current = NC_CURRENTS.get(abs(quarks[0])) if quarks[0] == -quarks[1] else None
        return None if current is None else (current, False)
    current = next((c for c, (up, down) in CC_PAIRS.items() if set(quarks) in ({up, -down}, {-up, down})), None)
    if len(charged) != 1 or neutrinos or current is None:
        return None
    hadronic_charge = sum(charge(p) for p in quarks)
    # a Majorana HNL decays to a neutral final state
    return (current, hadronic_charge > 0.0) if (charged[0] > 0) == (hadronic_charge > 0.0) else None


def hadronize(config, request):
    from .full_decays import subseed
    from .transition import boost, finish_partons, pythia_fraction

    if request['variation'] not in {'central', 'p-low', 'p-high', 'saturation-off'}:
        raise ValueError('Unsupported HNL hadronization variation')
    mass = float(request['mass'])

    if not np.isfinite(mass) or not HNL_MASS_GEV[0] <= mass <= HNL_MASS_GEV[1]:
        raise ValueError('HNL parent mass must lie in [%g, %g] GeV' % HNL_MASS_GEV)
    pdgs = tuple(request['pdgs'])
    spec = channel_spec(pdgs)

    if spec is None:
        raise ValueError('Unsupported signed HNL partonic current')
    current, conjugate = spec
    values = np.asarray(request['primary_events'], dtype=float)

    if not len(values):
        return []

    if values.ndim != 2 or values.shape[1] != 24 or not np.all(np.isfinite(values)):
        raise ValueError('HNL primaries must be finite 24-field three-body rows')
    particles = values.reshape(-1, 3, 8)

    if not np.all(particles[:, :, 5] == np.asarray(pdgs)):
        raise ValueError('HNL primary PDGs disagree with the selected signed row')

    if np.any(particles[:, :, 3:5] < 0):
        raise ValueError('HNL primary energy or mass is negative')

    if not np.allclose(particles[:, :, :4].sum(axis=1), [0., 0., 0., mass], rtol=2e-8, atol=2e-8):
        raise ValueError('HNL primary four-momentum does not close')
    shell = particles[:, :, 3]**2 - np.sum(particles[:, :, :3]**2, axis=2) - particles[:, :, 4]**2

    if np.max(np.abs(shell)) > 2e-7 * max(1., mass**2):
        raise ValueError('HNL primary particles are off shell')
    qcols = [i for i, p in enumerate(pdgs) if is_quark(p)]
    lcol = next(i for i, p in enumerate(pdgs) if not is_quark(p))
    frames = particles[:, qcols, :4].sum(axis=1)
    w = np.sqrt(np.maximum(frames[:, 3] ** 2 - (frames[:, :3] ** 2).sum(axis=1), 0.0))

    if current in PRIMARY_W_MIN and np.any(w < PRIMARY_W_MIN[current] - 2e-6):
        raise RuntimeError('HNL %s sampler produced W below the physical hadronic threshold %.6g GeV (minimum %.6g GeV)'
                           % (current, PRIMARY_W_MIN[current], float(w.min())))

    if np.any(w > mass - m0(pdgs[lcol]) + 2e-6):
        raise RuntimeError('HNL %s sampler exceeded its W endpoint' % current)
    supported = current in THRESHOLDS

    if not supported and (current not in PYTHIA_W_MIN or np.any(w < PYTHIA_W_MIN[current] - 2e-6)):
        raise ValueError('HNL heavy-flavour pair is below its hadronic threshold')
    fraction = pythia_fraction(w) if supported else np.ones(len(w))
    raw = np.random.default_rng(subseed(request['seed'], 'hadronization-transition')).random(len(w)) < fraction
    result = [None] * len(values)
    slots = np.flatnonzero(~raw)

    if len(slots):
        events = decays_at_W(current, list(w[slots]), request['seed'], request['variation'])
        for index, (slot, event) in enumerate(zip(slots, events)):
            frame = tuple(frames[slot])
            try:  # the final dilation pass moves last bits and admits no rest mass above W
                rest = _dilate(event, w[slot], final=True)
            except _Closed:
                rest = _regenerate(current, w[slot], request['seed'], index, request['variation'])
            hadrons = []
            for i in range(0, len(rest), 6):
                pdg = int(rest[i + 5])
                pdg = antiparticle(pdg) if conjugate else pdg
                hadrons.append([*map(float, _boost_into(frame, rest[i:i + 4])), rest[i + 4], float(pdg)])
            got = np.asarray(hadrons)[:, :4].sum(axis=0)
            if not np.allclose(got, frame, rtol=2e-8, atol=2e-9 * max(1.0, abs(float(frame[3])))):
                raise RuntimeError('HNL %s hadronic replacement violates four-momentum closure: got %r, expected %r'
                                   % (current, got.tolist(), list(map(float, frame))))
            result[slot] = [particles[slot, lcol, :6].tolist(), *hadrons]
    slots = np.flatnonzero(raw)

    if len(slots):
        pairs = [boost(particles[slot, qcols, :6], frames[slot], inverse=True) for slot in slots]
        hadrons = finish_partons(config, pairs, subseed(request['seed'], 'pythia'),
                                 CHARGED_CURRENT_SOURCE if current.startswith('CC') else NEUTRAL_CURRENT_SOURCE)
        for slot, event in zip(slots, hadrons):
            result[slot] = [particles[slot, lcol, :6].tolist(), *boost(event, frames[slot])]

    return result


def _boost_into(frame, p4):
    """Active boost of a rest-frame (px, py, pz, E) into the frame where `frame` has its momentum."""
    Px, Py, Pz, PE = frame
    M = np.sqrt(max(PE * PE - Px * Px - Py * Py - Pz * Pz, 1e-12))
    bx, by, bz = Px / PE, Py / PE, Pz / PE
    b2 = bx * bx + by * by + bz * bz

    if b2 < 1e-16:
        return p4
    gamma = PE / M
    px, py, pz, E = p4
    bp = bx * px + by * py + bz * pz
    fac = (gamma - 1.0) * bp / b2 + gamma * E

    return (px + fac * bx, py + fac * by, pz + fac * bz, gamma * (E + bp))


def _dilate(event, w, final=False):
    """Put a rest-frame six-field event on invariant mass W by a common 3-momentum dilation.

    Masses and identities are kept.  The router pass admits a rest-mass sum up to
    2e-12 max(1, W) above W and closes to that tolerance; the final pass admits none
    and closes to 2e-8 (momentum) and 2e-10 (energy) times max(1, W).
    """
    rows = [list(map(float, event[i:i + 6])) for i in range(0, len(event), 6)]
    if not rows:
        raise RuntimeError('empty HNL hadronic event')
    momenta = np.asarray([row[:3] for row in rows], dtype=float)
    masses = np.asarray([row[4] for row in rows], dtype=float)
    unit = max(1.0, w)
    rest = float(masses.sum())
    if rest > w + (0.0 if final else 2e-12 * unit):
        raise _Closed('HNL selected final state is closed at W: rest-mass sum %.12g > %.12g' % (rest, w))
    if final and len(rows) == 1:
        if not abs(rest - w) <= 1e-9:
            raise RuntimeError('one-particle residual state has mass %.12g, expected W %.12g' % (rest, w))
        return event
    p2 = np.sum(momenta * momenta, axis=1)
    if not np.any(p2) and rest < w:  # no momentum to dilate: scale overflow would give inf * 0 = NaN
        raise RuntimeError('HNL exact-W dilation of a zero-momentum state: rest-mass sum %.12g < W %.12g' % (rest, w))

    def energy(scale):
        return float(np.sqrt(masses * masses + scale * scale * p2).sum())
    lower, upper = 0.0, 1.0
    while energy(upper) < w:
        upper *= 2.0
    for _ in range(80):
        middle = 0.5 * (lower + upper)
        lower, upper = (middle, upper) if energy(middle) < w else (lower, middle)
    scale = 0.5 * (lower + upper)
    out, four = [], []
    for row, momentum, mass, q2 in zip(rows, momenta, masses, p2):
        scaled = scale * momentum
        four.append([float(scaled[0]), float(scaled[1]), float(scaled[2]), float(np.sqrt(mass * mass + scale * scale * q2))])
        out += [*four[-1], float(mass), row[5]]
    total = np.sum(np.asarray(four, dtype=float), axis=0)
    if not (np.linalg.norm(total[:3]) <= (2e-8 if final else 2e-12) * unit
            and abs(float(total[3]) - w) <= (2e-10 if final else 2e-12) * unit):  # NaN fails closed
        raise RuntimeError('HNL exact-W dilation failed four-momentum closure')
    return out


def _regenerate(current, w, seed, index, variation):
    """A final-pass closed event, regenerated by the router at exact W (eight seeded attempts, never vetoed)."""
    error = None

    for attempt in range(8):
        state = np.random.SeedSequence([int(seed), index, attempt, RETRY_STREAMS[current], 0x45585752]).generate_state(1, dtype=np.uint64)[0]
        retry = 1 + int(state % 899_900_000)
        try:
            return _dilate(decays_at_W(current, [float(w)], retry, variation, exact=True)[0], float(w), final=True)
        except Exception as exc:
            error = exc
    raise RuntimeError('HNL %s exact-W threshold retry failed at W=%.12g: seed=%d attempt=8/8; last error: %s'
                       % (current, w, retry, error)) from error


def _format(raw, seed):
    """Six-field events from dedicated (pdg, px, py, pz, E) lists, with the EventCalc pi0/K_S decays."""
    from .core.eventcalc import decay, decay_rng
    rng = decay_rng(seed)
    return [[value for pid, px, py, pz, e in event
             for row in decay([px, py, pz, e, np.sqrt(max(e * e - px * px - py * py - pz * pz, 0.0)), float(pid)], rng)
             for value in row] for event in raw]


def boundary_decays(current, categories, W_values, seed, exact=False, binary=None, xmldoc=None):
    """Fixed-edge categories (CC_ud, NC_ud) realized at each exact W."""
    from .hnl_backends import dedicated
    W_values = [float(w) for w in W_values]
    raw = dedicated.boundary_categories(current, categories, W_values, seed, exact, binary, xmldoc)
    return [_dilate(event, w) for event, w in zip(_format(raw, seed), W_values)]


def decays_at_W(current, W_values, seed, variation, exact=False):
    """One rest-frame six-field hadronic event per exact W (CC hadronic charge -1)."""
    from .core.eventcalc import decay_rng, six_field
    from .core.seeds import derive_event_seeds, derive_seam_seed
    from .hnl_backends import dedicated
    from .model1.sampler import HNL_DEPLOYMENTS, load_deployment, sample_hnl_exact_w
    values = [float(value) for value in W_values]
    if any(not np.isfinite(v) or v < THRESHOLDS[current] or v > 5.27 for v in values):
        raise ValueError('%s exact W lies outside physical support' % current)
    low, high = (load_deployment(HNL_DEPLOYMENTS[current]).activation_support_gev
                 if current in HNL_DEPLOYMENTS else (1.0, 0.0))
    edge, endpoint = dedicated.EDGES.get(current), ENDPOINTS.get(current)
    routes = {'dedicated': [], 'conditioned': [], 'projected': [], 'portable': []}
    for index, w in enumerate(values):
        if low <= w <= high:
            routes['portable'].append((index, w, None))
        elif current == 'CC_us' or (edge is not None and w <= edge):
            routes['dedicated'].append((index, w, None))
        elif endpoint is None:  # CC_cd, CC_cs
            routes['projected'].append((index, w, w))
        else:
            t = min(1.0, (w - edge) / (endpoint - edge))
            projected = t * t * (3.0 - 2.0 * t)
            seeds = derive_event_seeds(int(seed), index)
            if projected >= 1.0 or derive_seam_seed(seeds) / float(1 << 64) < projected:
                # component probabilities frozen at the endpoint, momenta generated at W
                routes['projected'].append((index, w, max(endpoint, w)))
            elif current == 'CC_ud':
                routes['conditioned'].append((index, w, _boundary_category(seeds, w)))
            else:  # NC_s: the edge composition, dilated to W below
                routes['dedicated'].append((index, edge, None))
    out = [None] * len(values)

    def fill(route, events):
        for (index, _, _), event in zip(routes[route], events):
            out[index] = event
    if routes['dedicated']:
        raw = dedicated.decays_at_W(current, [w for _, w, _ in routes['dedicated']], seed, exact)
        fill('dedicated', _format(raw, seed))
    if routes['conditioned']:
        fill('conditioned', boundary_decays(current, [c for _, _, c in routes['conditioned']],
                                            [w for _, w, _ in routes['conditioned']], seed, exact))
    if routes['projected']:  # the charm workers' +1 reference output is conjugated once to charge -1
        rng = decay_rng(seed)
        fill('projected', [_dilate(six_field(_projected(current, w, selection, seed, index, exact), rng,
                                             conjugate=current in ('CC_cd', 'CC_cs')), w)
                           for index, w, selection in routes['projected']])
    if routes['portable']:
        fill('portable', sample_hnl_exact_w(
            current, [w for _, w, _ in routes['portable']], seed=int(seed), variation=variation))
    for index, (event, w) in enumerate(zip(out, values)):
        try:
            out[index] = _dilate(event, w)
        except _Closed:
            if exact:
                raise
            # A binned state can close just above W: regenerate at exact W instead of vetoing it,
            # which would bias the category measure.
            retry = (int(seed) + 104729 * (index + 1)) % 899_900_000 or 1
            out[index] = decays_at_W(current, [w], retry, variation, exact=True)[0]
    return out


@lru_cache(maxsize=1)
def _release():
    """Tangent-preserving release of the CC_ud categories at 1.65 GeV over the 1.65-2.0 GeV window."""
    from .hnl_backends import dedicated
    from .hnl_backends.category_release import SimplexRelease
    masses, columns = dedicated._table('cc_channel_fractions.csv')
    previous, boundary = ({category: sum((columns[label][row] for label in labels), 0.0)
                          for category, labels in dedicated.CC_CATEGORIES.items()} for row in (-2, -1))
    slopes = {c: (boundary[c] - previous[c]) / (masses[-1] - masses[-2]) for c in boundary}
    boundary['remaining'] += 1.0 - sum(boundary.values())  # rounded CSV closure
    slopes['remaining'] -= sum(slopes.values())

    return SimplexRelease(1.65, 2.0, tuple(boundary.values()), tuple(slopes.values()))


def _boundary_category(seeds, w):
    from .core.seeds import derive_handoff_seed
    from .hnl_backends import dedicated
    retained = dict(zip(dedicated.CC_CATEGORIES, _release().retained_probabilities_at(w)))
    draw = derive_handoff_seed(seeds) / float(1 << 64) * sum(retained.values())
    positive = [(category, p) for category, p in retained.items() if p > 0.0]
    running = 0.0

    for index, (category, p) in enumerate(positive):
        running += p
        if draw < running or index + 1 == len(positive):
            return category


def _projected(current, w, selection, seed, index, exact):
    """One projected-string event at W, its component drawn from the tables at the selection W."""
    from .core.event_worker import (
        MULTIHADRON_CONTINUUM_STOP_MASS_GEV, EventRequest, PythiaComponentRoute, PythiaSubprocessRuntime)
    from .core.seeds import derive_event_seeds
    from .hnl_backends import dedicated
    source, table, components = PROJECTED[current]

    if current not in _RUNTIMES:
        # Stop mass on the light multihadron continuum only, not on charm routes.
        # A current averages the parent polarization, so it never carries its spin.
        routes = tuple(PythiaComponentRoute(cid, source, False, 'qq', runner,
                                            None if runner in ('cccd', 'cccs')
                                            else MULTIHADRON_CONTINUUM_STOP_MASS_GEV)
                       for runner, cid, _ in components)
        _RUNTIMES[current] = PythiaSubprocessRuntime(
            ROOT / 'cpp/exhad', dedicated.XMLDOC, maximum_condition_attempts=64, maximum_workers=3,
            routes=routes)
    probabilities = [1.0]

    if table is not None:
        masses, columns = dedicated._table(table)
        raw = [dedicated._interp(masses, columns[c[-1]], selection) for c in components]
        probabilities = [value / math.fsum(raw) for value in raw]
    seeds = derive_event_seeds(int(seed), index)
    draw, running = seeds.leaf / float(1 << 64), 0.0

    for k, (component, p) in enumerate(zip(components, probabilities)):
        running += p
        if draw < running or k + 1 == len(components):
            break
    gen_w = w

    if not exact:  # 25-MeV worker bins from 1.65 GeV; charm from threshold + 5 MeV, exact in the first 5 MeV
        charm = table is None
        first = THRESHOLDS[current] + 0.005 if charm else 1.65
        if not charm or w - THRESHOLDS[current] > 0.005:
            gen_w = min(w, first + math.floor(max(0.0, w - first) / 0.025) * 0.025)

    return _RUNTIMES[current].generate(EventRequest(gen_w, seeds.provider, component[1]))

