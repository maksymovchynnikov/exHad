"""Choose the complete decay palette; keep the hadronic sampler conditional.

Rate tables are evaluated once per mass, not once per generated decay.
No EventCalc program is imported; its rate tables are data/<model>/eventcalc_*.
"""
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from . import DATA
from .core.pdg import antiparticle, charge, is_lepton, m0
from .models import HNL_MASS_GEV, SCALARS

QUARK_MASSES = {1: .0047, 2: .0022, 3: .104, 4: 1.27, 5: 4.18}
STABLE = frozenset({11, 12, 13, 14, 16, 22, 211, 321, 130, 2112, 2212})  # |PDG| left undecayed
# Lightest Pythia 8.317 hadron carrying each parton flavour: a colour-singlet
# pair hadronizes only above twice this mass (open charm D0, open bottom B+).
LIGHTEST_HADRON = {21: .13498, 1: .13498, 2: .13498, 3: .49368, 4: 1.86486, 5: 5.27925}
LEPTON_PAIRS = ((11, 'e+e-'), (13, 'mu+mu-'), (15, 'tau+tau-'))


@lru_cache(maxsize=16)
def table(path):
    """Rate-table rows; their node lists become float arrays once, not at every evaluation."""
    return [[*row[:2], *(np.asarray(x, dtype=float) if isinstance(x, list) else x for x in row[2:])]
            for row in json.loads(Path(path).read_text())]


def interpolate(points, mass, threshold=None):
    """Linear in mass between rate nodes, without extrapolation.

    With a physical threshold the rate is zero at and below it and linear from
    zero at the threshold to the first node above it; the nodes are unchanged.
    """
    values = np.asarray(points, dtype=float)
    if mass < values[0, 0] - 1e-12 or mass > values[-1, 0] + 1e-12:
        raise ValueError('Mass outside the decay-rate table; extrapolation is disabled')
    if threshold is not None:
        if mass <= threshold:
            return 0.
        node, rate = values[np.searchsorted(values[:, 0], threshold, 'right')]
        if mass < node:
            return float(rate * (mass - threshold) / (node - threshold))
    return float(np.interp(mass, values[:, 0], values[:, 1]))


def row_threshold(pdgs):
    """Lightest hadronic final state of a supplied alp-fermion or scalar row.

    A parton pair needs two of the lightest hadrons of its flavour, so b-bbar opens
    at 2 m(B+) = 10.5585 GeV although the supplied rate is positive from 10.543 GeV.
    With this threshold interpolate() gives zero at and below it and is linear
    from it to the first node above it.
    """
    return sum(LIGHTEST_HADRON[abs(p)] if abs(p) in LIGHTEST_HADRON else m0(p) for p in pdgs)


def normalized(values):
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0) or values.sum() <= 0:
        raise ValueError('Invalid decay-channel probabilities')
    if not math.isclose(math.fsum(values), 1., abs_tol=2e-8, rel_tol=0):
        raise ValueError(f'Decay channels do not exhaust the total width: sum={math.fsum(values):.12g}')
    return values / values.sum()  # Roundoff only; never repair missing modes.


def subseed(seed, label):
    return int.from_bytes(hashlib.sha256(f'exhad-full-v1:{seed}:{label}'.encode()).digest()[:8], 'big')


def boson_rates(model, mass, variation):
    """Return labels, primary PDGs, and absolute branching fractions.

    Below the start mass (exclusive.START) the hadronic slot is the sum of the exclusive rows;
    the dark-photon, ALP and scalar lepton and photon rows come from the same table vector.
    """
    from . import exclusive
    if model not in exclusive.START:
        raise ValueError('Unknown complete-decay model: ' + model)
    low, high = exclusive.table_start(model), 63. if model in SCALARS else 5.
    if not low <= mass <= high:
        raise ValueError(f'{model} full decays require {low:g} <= mass <= {high:g} GeV')
    below = mass < exclusive.START[model]
    if below:
        exclusive.check_variation(model, mass, variation)
    if below and model != 'b-l':
        catalog, values = exclusive.rows(model), exclusive.rates(model, mass)
        hadronic = math.fsum(v for row, v in zip(catalog, values) if row.has_hadrons)
        nonhad = [(row, v) for row, v in zip(catalog, values) if not row.has_hadrons]
        pairs = dict(LEPTON_PAIRS) if model == 'dark-photon' else {}  # the lepton labels and PDG order above 1.70 GeV
        labels = [pairs.get(abs(row.pdgs[0]), row.label) for row, _ in nonhad]
        ids = [list(row.pdgs) for row, _ in nonhad]
        widths = [v for _, v in nonhad]
    elif model == 'alp-fermion' or model in SCALARS:
        relative = ('alp-fermion/eventcalc_branching_ratios.json' if model == 'alp-fermion' else
                    f'scalar/eventcalc_branching_ratios_{SCALARS[model]}.json')
        rows = table(str(DATA / relative))
        if model == 'alp-fermion':
            # EventCalc's ALP gap rule: on [1.911 GeV, first partonic node) every row is held at that node.
            nodes = rows[0][2][:, 0]
            mass = max(mass, float(nodes[np.searchsorted(nodes, exclusive.START[model])]))
        values = [interpolate(r[2], mass) for r in rows]
        nonhad = [i for i, r in enumerate(rows)
                  if all(is_lepton(p) or int(p) in {22, -999} for p in r[1])]
        had = [i for i in range(len(rows)) if i not in nonhad]
        opened = [interpolate(r[2], mass, row_threshold([int(p) for p in r[1] if int(p) != -999])) for r in rows]
        ids = [[int(p) for p in rows[i][1] if int(p) != -999] for i in nonhad]
        labels = [rows[i][0] for i in nonhad]
        widths = [opened[i] for i in nonhad]
        hadronic = math.fsum(opened[i] for i in had)
        if model in SCALARS and abs(math.fsum(values) - 1.) > 1e-12:
            # The scalar inputs have no S -> gamma gamma channel: the listed channels must exhaust the width.
            raise ValueError('Scalar input branching fractions must sum to one')
        # Rate interpolated below a closed threshold is no width: the total is the open rates.
        total = 1. - ((math.fsum(values[i] for i in nonhad) - math.fsum(widths))
                      + (math.fsum(values[i] for i in had) - hadronic))
        hadronic, widths = hadronic / total, [w / total for w in widths]
    else:
        from .registry import b_l, dark_photon
        if model == 'dark-photon':
            from .dark_photon.widths import unit_current_inclusive_uds_width
            _, charm = dark_photon(None if variation == 'central' else variation)
            hadronic = unit_current_inclusive_uds_width(mass)
            if mass >= charm.owner.card.domain_gev[0]:
                hadronic += charm.owner.closure_at(mass).inclusive_charm_width
        elif below:
            from .b_l.low_mass import hadronic_width
            hadronic = hadronic_width(mass)
        else:
            hadronic = b_l().rate_point(mass, variation=variation).inclusive_width_per_g_b_squared_gev
        labels, ids, widths = [], [], []
        for pid, label in LEPTON_PAIRS:
            x = (m0(pid) / mass)**2
            labels.append(label)
            ids.append([-pid, pid])
            widths.append(mass / (12 * math.pi) * (1 + 2 * x) * math.sqrt(1 - 4 * x) if x < .25 else 0.)
        if model == 'b-l':
            # Minimal B-L benchmark: three active left-handed neutrinos;
            # additional on-shell right-handed neutrinos are not assumed.
            for pid in (12, 14, 16):
                labels.append(f'nu{pid}-antinu{pid}')
                ids.append([pid, -pid])
                widths.append(mass / (24 * math.pi))
        total = math.fsum([hadronic, *widths])
        hadronic /= total
        widths = [w / total for w in widths]
    return ['hadronic', *labels], [None, *ids], normalized([hadronic, *widths])


@lru_cache(maxsize=1)
def hnl_tables():
    directory = DATA / 'hnl'
    return table(str(directory / 'eventcalc_branching_ratios.json')), np.loadtxt(directory / 'eventcalc_total_widths.dat')


def hnl_rates(mass, mixing):
    """Select pure mixing and signed row jointly, using |U_alpha|^2 Gamma_alpha.

    Every row is zero at and below its threshold (quarks at the partonic masses FICTITIOUS) and linear
    to its first node above. The channels are those of the tables, which give the total width.
    """
    from .core.kinematics import FICTITIOUS
    if not HNL_MASS_GEV[0] <= mass <= HNL_MASS_GEV[1]:
        raise ValueError('HNL full decays require %g <= mass <= %g GeV' % HNL_MASS_GEV)
    mix = np.asarray(mixing, dtype=float)
    if mix.shape != (3,) or not np.all(np.isfinite(mix)) or np.any(mix < 0) or mix.sum() <= 0:
        raise ValueError('HNL requires three nonnegative squared mixings, with a positive sum')
    rows, widths = hnl_tables()
    tabulated_total = np.array([np.interp(mass, widths[:, 0], widths[:, i + 1]) for i in range(3)])
    fractions = mix * tabulated_total
    fractions /= fractions.sum()
    labels, ids, expressions, probabilities = [], [], [], []
    closed = 0.  # Rate interpolated below an exclusive threshold: no width of any channel.
    for alpha in range(3):
        if fractions[alpha] == 0:
            continue
        tabulated = [interpolate(row[2 + alpha], mass) for row in rows]
        invisible = 1. - math.fsum(tabulated)  # Gamma_3nu completes the tabulated partition.
        if invisible < -2e-8:
            raise ValueError('HNL visible rates exceed the total width')
        for row, br in zip(rows, tabulated):
            pdgs = [int(p) for p in row[1] if int(p) != -999]
            if br > 0:
                opened = interpolate(row[2 + alpha], mass, sum(
                    FICTITIOUS[abs(p)] if abs(p) in QUARK_MASSES else m0(p) for p in pdgs))
                closed += fractions[alpha] * (br - opened)
                br = opened
            if br <= 0:
                continue
            # Neutral-current rows use nu_e as a flavor placeholder and include
            # the Majorana charge conjugate. Resolve both in the output record.
            nc = row[0].endswith('v') and not row[0].startswith(('emu', 'etau', 'mutau'))
            signs = (1, -1) if nc else (1,)
            if row[0].startswith(('emu', 'etau', 'mutau')) and abs(pdgs[2]) == 11 + 2 * alpha:
                # Mixing of the second lepton's flavour: N -> l2 W, W -> l1 nubar_l1.
                pdgs[1] = -int(math.copysign(abs(pdgs[0]) + 1, pdgs[0]))
            for sign in signs:
                selected = []
                for p in pdgs:
                    if nc and abs(p) == 12:
                        selected.append(sign * (12 + 2 * alpha))
                    else:
                        selected.append(antiparticle(p) if sign < 0 else p)
                labels.append(row[0])
                ids.append(selected)
                expressions.append(row[5 + alpha])
                probabilities.append(fractions[alpha] * br / len(signs))
        # Gamma_3nu is already in the lifetime table. Each beta contributes
        # 1+delta(alpha,beta); the Majorana conjugates have equal widths.
        for beta in range(3):
            for sign in (1, -1):
                labels.append('3nu')
                ids.append([sign * (12 + 2 * alpha), sign * (12 + 2 * beta), -sign * (12 + 2 * beta)])
                expressions.append('E3*(mLLP-2*E3)')
                probabilities.append(fractions[alpha] * max(0., invisible) * (1 + (alpha == beta)) / 8)
    return labels, ids, expressions, normalized(np.asarray(probabilities) / (1. - closed))


def primary(mass, pdgs, count, seed, expression='1.', quark_masses=QUARK_MASSES):
    from .core import kinematics
    charges, stabilities = [charge(p) for p in pdgs], [0] * len(pdgs)
    masses = [quark_masses.get(abs(p), m0(p)) for p in pdgs]
    kinematics.seed(seed % 899900000 + 1)

    if len(pdgs) == 2:
        return kinematics.two_body(mass, count, *masses, *pdgs, *charges, *stabilities)

    if len(pdgs) == 3:
        return kinematics.three_body(mass, count, pdgs, masses, charges, stabilities, expression)

    if len(pdgs) >= 4 and expression in kinematics.FLAT:
        return kinematics.n_body(mass, count, pdgs, masses, charges, stabilities)
    raise ValueError('Full-decay primary row must have two or three particles, or at least four with flat phase space')


def check_complete(events, count, mass):
    """Raise unless there are count events, each conserving the four-momentum of the parent at rest."""
    if len(events) != count:
        raise RuntimeError('Full-decay channel returned the wrong number of events')
    for event in events:
        if not np.allclose(np.asarray(event)[:, :4].sum(axis=0), [0., 0., 0., mass], atol=2e-7, rtol=2e-7):
            raise RuntimeError('Full decay does not conserve four-momentum')


def realize_rows(config, mass, rows, counts, seed, terminal='pythia', request=None):
    """Complete rest-frame events {key: events} of explicit decay rows {key: (pdgs, |M|^2 expression)}.

    Row key k draws counts[k] primaries with seed subseed(seed, k). terminal='pythia' decays every
    unstable daughter with Pythia; terminal='matched' gives the row's other unstable particles the
    matched-event convention (matched_terminal). Taus keep full Pythia decays under both terminals,
    so rows whose only unstable particles are taus are identical under the two. HNL partonic rows
    hadronize with the full-decay request.
    """
    from .secondary import decayer
    if terminal not in {'pythia', 'matched'}:
        raise ValueError("terminal must be 'pythia' or 'matched'")
    realized = {}
    for key, (pdgs, expression) in rows.items():
        channel_seed = subseed(seed, key)
        generated = primary(mass, pdgs, counts[key], channel_seed, expression)
        if any(abs(p) in QUARK_MASSES for p in pdgs):
            from .hnl import hadronize
            events = hadronize(config, dict(request, pdgs=pdgs,
                primary_events=generated.tolist(), seed=channel_seed))
            if any(abs(p) == 15 for p in pdgs):
                # hadronize_hnl deliberately preserves the spectator.
                # A complete decay sample must subsequently decay a tau,
                # without sending already-decayed hadrons through again.
                daughters = decayer(config['xmldoc']).finish(
                    [[event[0]] for event in events], subseed(channel_seed, 'spectator-tau'))
                events = [[*tau, *event[1:]] for tau, event in zip(daughters, events)]
        else:
            events = generated.reshape(counts[key], -1, 8)[:, :, :6].tolist()
            if terminal == 'matched' and {abs(p) for p in pdgs} - STABLE - {15}:
                events = matched_terminal(config['xmldoc'], events, pdgs, channel_seed)
            elif any(abs(p) not in STABLE for p in pdgs):
                events = decayer(config['xmldoc']).finish(events, channel_seed)
        check_complete(events, counts[key], mass)
        realized[key] = events
    return realized


def matched_terminal(xml, events, pdgs, seed):
    """Six-field events of one explicit row in the matched-event terminal convention.

    The row's particles other than taus decay in Pythia with pi0 and K_S kept stable, which then take
    the matched-event pi0 -> gamma gamma and K_S decays; rows of only {e, nu, mu, gamma, pi+-, K+-, K_L,
    p, n, pi0, K_S} skip Pythia. Taus keep full Pythia decays with seed subseed(seed, 'spectator-tau'),
    as the tau spectators of partonic rows do; their daughters follow the other particles.
    """
    from .core.eventcalc import decay, decay_rng
    from .secondary import decayer
    tau = [abs(p) == 15 for p in pdgs]
    taus = [[p for p, is_tau in zip(event, tau) if is_tau] for event in events]
    events = [[p for p, is_tau in zip(event, tau) if not is_tau] for event in events]
    if any(abs(p) not in STABLE | {111, 310} for p, is_tau in zip(pdgs, tau) if not is_tau):
        events = decayer(xml, (111, 310)).finish(events, seed)
    rng = decay_rng(seed)
    events = [[q for p in event for q in decay(p, rng)] for event in events]
    if any(tau):
        taus = decayer(xml).finish(taus, subseed(seed, 'spectator-tau'))
    return [[*rest, *daughters] for rest, daughters in zip(events, taus)]


def sample_all(config, request, sample_hadronic):
    model, mass, count, seed = (request[k] for k in ('model', 'mass', 'events', 'seed'))
    weighted = request.get('weighted', False)

    if weighted and model != 'alp-fermion':
        raise ValueError('Weighted full-decay sampling supports only alp-fermion')

    if model == 'hnl':
        labels, ids, expressions, probs = hnl_rates(mass, request.get('mixing'))
        if not request.get('all_decays', False):
            # generate(): the semileptonic decays, the rows with a hadron or a quark.
            semileptonic = np.array([not all(abs(p) in {11, 12, 13, 14, 15, 16, 22} for p in pdgs) for pdgs in ids])
            if not probs[semileptonic].sum() > 0:
                raise ValueError(f'hnl has no open semileptonic channel at {mass:g} GeV for these mixings; use generate_all')
            probs = np.where(semileptonic, probs, 0.) / probs[semileptonic].sum()
    else:
        labels, ids, probs = boson_rates(model, mass, request['variation'])
        expressions = ['1.'] * len(labels)
    rng = np.random.default_rng(subseed(seed, 'channels'))
    selected = rng.choice(len(probs), size=count, p=probs)
    result = dict(events=[None] * count, channel_labels=[labels[i] for i in selected],
        decay_scope='all', raw_weights=[1.] * count, normalization_groups=['external'] * count,
        family_labels=[labels[i] for i in selected])
    slots = {int(i): np.flatnonzero(selected == i) for i in np.unique(selected)}

    for i in (i for i in slots if ids[i] is None):  # the hadronic channel is the first
        sub = dict(request, all_decays=False, events=len(slots[i]), seed=subseed(seed, i))
        sampled = sample_hadronic(config, sub)
        events = sampled['events'] if weighted else sampled
        if weighted:
            for key in ('raw_weights', 'normalization_groups', 'family_labels'):
                for slot, value in zip(slots[i], sampled[key]):
                    result[key][slot] = value
            result.update({key: sampled[key] for key in ('weight_convention', 'weight_floor_fraction')})
        check_complete(events, len(slots[i]), mass)
        for slot, event in zip(slots[i], events):
            result['events'][slot] = event
    explicit = {i: (ids[i], expressions[i]) for i in slots if ids[i] is not None}
    realized = realize_rows(config, mass, explicit, {i: len(slots[i]) for i in explicit}, seed, request=request)

    for i, events in realized.items():
        for slot, event in zip(slots[i], events):
            result['events'][slot] = event

    if weighted:
        result.setdefault('weight_convention', 'self-normalized-within-fragmentation; external-unit-weight')
        result.setdefault('weight_floor_fraction', request.get('weight_floor_fraction', 0.))
    else:
        for key in ('raw_weights', 'normalization_groups', 'family_labels'):
            result.pop(key)

    return result
