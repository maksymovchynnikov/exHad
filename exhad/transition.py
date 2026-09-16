"""Rate-preserving mixture of complete generators on 4 < W < 5 GeV.

W is the boson mass or the HNL quark-pair invariant mass. A quintic
smoothstep makes the mixture coefficient, its slope and curvature join at
both ends. No momenta are interpolated, and no partial width is changed.
The alp-fermion (to 5 GeV) and the scalars (to 63 GeV) shower their supplied rows.
"""
from functools import lru_cache
import numpy as np
from . import DATA, ROOT
from .core.pdg import is_lepton
from .full_decays import LIGHTEST_HADRON, QUARK_MASSES, STABLE, interpolate, primary, row_threshold, subseed, table
from .models import MODEL_SOURCE, SCALARS
from .secondary import Decayer, decayer

# Seed labels of the supplied-row choice; they are part of the random-stream identity.
ALP_FERMION_SOURCE_SEED_LABEL, SCALAR_SOURCE_SEED_LABEL = 'alp-source', 'scalar-source'


def pythia_fraction(w):
    x = np.clip(np.asarray(w, dtype=float) - 4., 0., 1.)
    return x * x * x * (10. + x * (-15. + 6. * x))


@lru_cache(maxsize=4)
def shower(xml, source):
    return Decayer(ROOT / '.runtime/exhad-shower', xml, *source.arguments)


def finish_partons(config, events, seed, source):
    """Shower and hadronize parton pairs of a source with these quantum numbers."""
    return shower(config['xmldoc'], source).finish(events, seed)


def boost(particles, frame, inverse=False):
    """Lorentz boost without changing masses or particle identifiers."""
    rows = np.asarray(particles, dtype=float).copy()
    frame = np.asarray(frame, dtype=float)
    beta = frame[:3] / frame[3]

    if inverse:
        beta = -beta
    b2 = np.dot(beta, beta)

    if b2 == 0:
        return rows.tolist()
    gamma = 1 / np.sqrt(1 - b2)
    dot = rows[:, :3] @ beta
    rows[:, :3] += (((gamma - 1) * dot / b2 + gamma * rows[:, 3])[:, None] * beta)
    rows[:, 3] = gamma * (rows[:, 3] + dot)

    return rows.tolist()


def showered(config, model, mass, count, seed):
    """Select supplied hadronic rows, then generate each at its fixed flavour."""
    alp_fermion = model == 'alp-fermion'
    source = 'alp-fermion/eventcalc_branching_ratios.json' if alp_fermion else f'scalar/eventcalc_branching_ratios_{SCALARS[model]}.json'
    rows = [(r[0], [int(p) for p in r[1] if int(p) != -999], r[2]) for r in table(str(DATA / source))]
    had = [r for r in rows if not all(is_lepton(p) or p == 22 for p in r[1])]
    rates = np.array([interpolate(values, mass, row_threshold(ids)) for _, ids, values in had])
    chosen = np.random.default_rng(subseed(seed, ALP_FERMION_SOURCE_SEED_LABEL if alp_fermion else SCALAR_SOURCE_SEED_LABEL)).choice(len(had), size=count, p=rates / rates.sum())
    result = [None] * count

    for i in np.unique(chosen):
        slots = np.flatnonzero(chosen == i)
        ids = had[i][1]
        if len(ids) != 2:
            raise ValueError('Unexpected high-mass exclusive row')
        key = subseed(seed, int(i))
        # The scalar hands Pythia its own constituent parton masses (Pythia m0).
        events = primary(mass, ids, len(slots), key, quark_masses=QUARK_MASSES if alp_fermion else {})
        events = events.reshape(len(slots), 2, 8)[:, :, :6].tolist()
        if any(abs(p) in LIGHTEST_HADRON for p in ids):
            events = finish_partons(config, events, key, MODEL_SOURCE[model])
        elif any(abs(p) not in STABLE for p in ids):
            events = decayer(config['xmldoc']).finish(events, key)
        for slot, event in zip(slots, events):
            result[slot] = event

    return result


def sample(config, request, matched):
    model, mass, count, seed = (request[k] for k in ('model', 'mass', 'events', 'seed'))
    high = 5. if model == 'alp-fermion' else 63.

    if not 4. < mass <= high:
        raise ValueError(f'{model} transition requires 4 < mass <= {high:g} GeV')

    if request['variation'] not in {'central', 'p-low', 'p-high', 'saturation-off'}:
        raise ValueError(f'Unsupported {model} hadronization variation')
    rng = np.random.default_rng(subseed(seed, 'hadronization-transition'))
    raw = rng.random(count) < pythia_fraction(mass)
    result = [None] * count
    metadata = dict(raw_weights=[1.] * count, normalization_groups=['external'] * count,
                    family_labels=['pythia'] * count)

    for take_raw in (False, True):
        slots = np.flatnonzero(raw == take_raw)
        if not len(slots):
            continue
        key = subseed(seed, 'pythia' if take_raw else 'matched')
        if take_raw:
            events = showered(config, model, mass, len(slots), key)
        else:
            answer = matched(config, dict(request, events=len(slots), seed=key))
            events = answer['events'] if request.get('weighted') else answer
            if request.get('weighted'):
                for name in metadata:
                    for slot, value in zip(slots, answer[name]):
                        metadata[name][slot] = value
        for slot, event in zip(slots, events):
            result[slot] = event

    if not request.get('weighted'):
        return result

    return dict(events=result, **metadata,
        weight_convention='self-normalized-within-fragmentation; external-unit-weight',
        weight_floor_fraction=request.get('weight_floor_fraction', 0.))
