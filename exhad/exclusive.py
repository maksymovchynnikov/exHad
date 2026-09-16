"""Explicit rows of a model's EventCalc-format decay table.

Two uses share one row catalog:

* host-count realization (generate_rows): the host owns the rates, a row is realized at any
  mass where its particles are open, whatever its tabulated rate there;
* the portal's own decays below its start mass (START): exclusive rates, one row drawn per
  event, realized with the matched terminal convention (pi0 -> gamma gamma, K_S -> pi pi as in
  matched events above the start mass). Model-1, the matched dark-photon edge and the B-L
  accelerator own every mass at and above the start mass.

Threshold rule (every explicit row of every portal): the rate is zero at and below the sum of
the generator's particle masses and linear from zero there to the first node above; the
removed rate is renormalized away. Partonic (Jets) rows carry no explicit final state.

Variation rule (check_variation, every portal and every variation name): a model variation
varies the matched inputs, which start at START. Below the start mass no variant of the
exclusive rows exists, so every variation other than 'central' is rejected there.

The rule also covers a defect of the scalar inputs: their KLKL and KSKS rows carry positive
nodes on 0.988-0.995 GeV, below 2 m_K0 = 0.99522 GeV (up to 0.105 per row for scalar-upper),
because the tabulated nodes use the charged-kaon threshold. Those rates are zeroed and
renormalized away here; the tables themselves are not edited.

The kinematics change at the start masses by construction: below, one tabulated channel with
flat or |M|^2-weighted phase space and pole masses for the omega and phi; above, a fragmented
Pythia state with resonance line shapes. exHad accepts that seam at 1.70, 1.911 and
2.0 GeV and does not blend the two descriptions.
"""
from functools import lru_cache
import math
from types import SimpleNamespace
import numpy as np
from . import DATA
from .core.kinematics import FICTITIOUS
from .core.pdg import m0
from .full_decays import hnl_rates, interpolate, realize_rows, subseed, table
from .models import SCALARS, eventcalc_tables

# Start masses (models.model_info support_gev): exclusive rows below, matched generation from here.
START = {'dark-photon': 1.7, 'alp-fermion': 1.911, 'b-l': 2., **dict.fromkeys(SCALARS, 2.)}
LEPTONIC = frozenset({11, 12, 13, 14, 15, 16, 22})
HBARC_GEV_M = 1.973269804e-16


def start_mass(model):
    """Start mass of a table portal (0 for HNL, whose rows join inside its table)."""
    return START.get(model, 0.)


@lru_cache(maxsize=None)
def rows(model):
    """Row catalog of a public model: label, PDGs, node array, |M|^2 expression, partonic, has_hadrons, threshold."""
    catalog = []

    for row in table(str(DATA / eventcalc_tables(model)['decay'])):
        pdgs = tuple(int(p) for p in row[1] if int(p) != -999)
        partonic = any(abs(p) in FICTITIOUS for p in pdgs)
        catalog.append(SimpleNamespace(
            label=row[0], pdgs=pdgs, values=row[2], partonic=partonic, expression=row[3],
            has_hadrons=not all(abs(p) in LEPTONIC for p in pdgs),
            threshold=None if partonic else math.fsum(m0(p) for p in pdgs)))

    return tuple(catalog)


def hadron_row_threshold(model):
    """Lowest threshold of an explicit row with hadrons: exclusive generation is open above it."""
    return min(row.threshold for row in rows(model) if row.has_hadrons and not row.partonic)


def table_start(model):
    return float(rows(model)[0].values[0, 0])


def _held_node(model, mass):
    """The last table node below the start mass: the gap interval (node, start) has its own rule."""
    nodes = rows(model)[0].values[:, 0]
    return float(nodes[nodes < START[model]][-1])


def rates(model, mass):
    """Branching ratios of every table row below the start mass (dark photon, ALP, scalars)."""
    mass = float(mass)
    low, start = table_start(model), START[model]

    if not low <= mass < start:
        raise ValueError(f'{model} exclusive rates cover {low:g} <= mass < {start:g} GeV')
    catalog, edge = rows(model), _held_node(model, mass)
    gap = edge < mass
    # Gap rules. ALP (EventCalc's): every row is held at the last exclusive node, then the nucleon bridge.
    # Dark photon: every row is held at 1.699 GeV. The table's 1.70 node carries the partonic rows with
    # Gamma_had/Gamma_mumu = 1.98 and sums to 0.952, while the exclusive rows at 1.699 give 2.183 and the
    # matched width at 1.70 GeV 2.182; interpolating towards that node would open a 4.6% hadronic step.
    held = gap and model in ('alp-fermion', 'dark-photon')
    values = np.array([interpolate(row.values, edge if held else mass, row.threshold) for row in catalog])

    if gap and model == 'alp-fermion':
        values = alp_nucleon_bridge(values, mass)
    elif gap and model != 'dark-photon':
        values = scalar_endpoint(model, values, mass, edge)

    if np.any(values[[row.partonic for row in catalog]] != 0):
        raise ValueError(f'{model} partonic rows are open below the start mass')

    return values / math.fsum(values)


@lru_cache(maxsize=1)
def _alp_ctau():
    return np.loadtxt(DATA / eventcalc_tables('alp-fermion')['ctau'])


def alp_nucleon_bridge(values, mass):
    """EventCalc's fermionic-ALP gap rule between the last exclusive and the first partonic node.

    Every row is held at the node on its side of the start mass (values). p pbar and n nbar
    are exact two-body widths across the gap: their BR/ctau follows the monotone (Fritsch-Carlson)
    cubic Hermite segment of the one-sided table tangents at both nodes, times ctau(m); the other
    hadronic rows are rescaled so that the held hadronic total is kept.
    """
    catalog, ctau = rows('alp-fermion'), _alp_ctau()
    nodes = catalog[0].values[:, 0]
    right = int(np.searchsorted(nodes, START['alp-fermion']))
    left, start, end = right - 1, nodes[right - 1], nodes[right]
    length = end - start
    result = np.array(values, dtype=float)
    hadronic = [i for i, row in enumerate(catalog) if row.has_hadrons]
    persistent = [i for i, row in enumerate(catalog) if row.label in ('nnbar', 'ppbar')]
    target = math.fsum(result[hadronic])
    lifetime = lambda m: float(np.interp(m, ctau[:, 0], ctau[:, 1]))

    def endpoint(br, node, lower):
        """BR/ctau at a node and its derivative from the one-sided (lower or upper) table segments."""
        a, b = (node - 1, node) if lower else (node, node + 1)
        span = nodes[b] - nodes[a]
        slope = (br[b] - br[a]) / span
        ctau_slope = (lifetime(nodes[b]) - lifetime(nodes[a])) / span
        c = lifetime(nodes[node])
        return br[node] / c, (slope * c - br[node] * ctau_slope) / (c * c)

    for i in persistent:
        br = catalog[i].values[:, 1]
        (y0, d0), (y1, d1) = endpoint(br, left, True), endpoint(br, right, False)
        secant = (y1 - y0) / length
        if secant == 0.:
            d0 = d1 = 0.
        else:
            d0, d1 = (0. if d * secant <= 0. else d for d in (d0, d1))
            norm = math.hypot(d0 / secant, d1 / secant)
            if norm > 3.:
                d0, d1 = 3. * d0 / norm, 3. * d1 / norm
        t = (mass - start) / length
        reduced_width = ((2 * t**3 - 3 * t**2 + 1) * y0 + (t**3 - 2 * t**2 + t) * length * d0
                         + (-2 * t**3 + 3 * t**2) * y1 + (t**3 - t**2) * length * d1)
        if reduced_width < 0.:
            raise ValueError('ALP nucleon-width bridge became negative')
        result[i] = reduced_width * lifetime(mass)
    active = [i for i in hadronic if i not in persistent]
    scale = (target - math.fsum(result[persistent])) / math.fsum(result[active])
    if not scale > 0.:
        raise ValueError('ALP nucleon bridge has no positive complement')
    result[active] *= scale
    return result


def scalar_endpoint(model, values, mass, edge):
    """EventCalc's scalar-endpoint-amplitude-v1 on the last table interval below 2 GeV.

    Each two-meson amplitude is held at the last exclusive node: BR(m) = BR(m0) Gamma(m0)/Gamma(m)
    (m0/m) beta(m)/beta(m0), with Gamma = hbar c/ctau linear between nodes. Nucleon and leptonic
    rows keep their interpolated rates; the remaining hadronic rate goes to the four-pion rows in
    their m0 ratios, so the whole light-hadronic width is four-pion as m -> 2 GeV.
    """
    catalog = rows(model)
    lifetime = _scalar_ctau(model)
    width = lambda m: float(np.interp(m, lifetime[:, 0], HBARC_GEV_M / lifetime[:, 1]))
    at_edge = [interpolate(row.values, edge, row.threshold) for row in catalog]
    result = np.array(values, dtype=float)
    hadronic = [i for i, row in enumerate(catalog) if row.has_hadrons]
    mesons = [i for i in hadronic if not catalog[i].partonic and len(catalog[i].pdgs) == 2 and all(abs(p) < 1000 for p in catalog[i].pdgs)]
    four_pion = [i for i in hadronic if len(catalog[i].pdgs) == 4 and all(abs(p) in (111, 211) for p in catalog[i].pdgs)]
    target = math.fsum(result[hadronic])
    for i in mesons:
        daughter = m0(catalog[i].pdgs[0])
        beta = lambda m: math.sqrt(1 - 4 * daughter * daughter / (m * m))
        result[i] = at_edge[i] * width(edge) / width(mass) * (edge / mass) * beta(mass) / beta(edge)
    result[[row.partonic for row in catalog]] = 0.
    residual = target - math.fsum(result[i] for i in hadronic if i not in four_pion)
    if residual < 0:
        raise ValueError('continued two-meson rates exceed the inclusive hadronic width')
    norm = math.fsum(at_edge[i] for i in four_pion)
    for i in four_pion:
        result[i] = residual * at_edge[i] / norm
    return result


@lru_cache(maxsize=None)
def _scalar_ctau(model):
    import json
    return np.asarray(json.loads((DATA / eventcalc_tables(model)['ctau']).read_text()), dtype=float)


def check_variation(model, mass, variation):
    """The variation rule of the module docstring: a deployed variation at or above START(model),
    and only 'central' below it, where the exclusive rows have no variant."""

    if variation == 'central':
        return

    if float(mass) < START[model]:
        raise ValueError(f'{model} has no {variation!r} variation at {float(mass):g} GeV: its inputs below the '
                         f'start mass {START[model]:g} GeV have no variant')

    if model == 'dark-photon':
        from .registry import dark_photon
        dark_photon(variation)
        return
    from .model1.sampler import load_deployment

    if variation not in load_deployment(model).supported_variations:
        raise ValueError(f'Unknown {model} variation {variation!r}')


def sample_hadronic(config, request):
    """Hadronic decays below the start mass: one exclusive row per event (weighted ALP: unit external weights)."""
    model, mass, count, seed = request['model'], float(request['mass']), request['events'], request['seed']
    check_variation(model, mass, request['variation'])
    low = table_start(model)

    if mass < low:
        raise ValueError(f'{model} decays start at the table mass {low:g} GeV')

    if model == 'b-l':
        from .b_l.low_mass import sample
        return sample(config, mass, count, seed)
    catalog, probabilities = rows(model), rates(model, mass)
    chosen_rows = [i for i, row in enumerate(catalog) if row.has_hadrons and probabilities[i] > 0]
    total = math.fsum(probabilities[chosen_rows])

    if not total > 0:
        raise ValueError(f'{model} has no open hadronic channel at {mass:g} GeV; use generate_all')
    chosen = np.random.default_rng(subseed(seed, 'exclusive-rows')).choice(
        len(chosen_rows), count, p=probabilities[chosen_rows] / total)
    labels = [catalog[chosen_rows[j]].label for j in chosen]
    events = realize_labels(config, model, mass, labels, seed)

    if not request.get('weighted', False):
        return events

    return dict(events=events, raw_weights=[1.] * count, normalization_groups=['external'] * count,
                family_labels=labels, weight_convention='self-normalized-within-fragmentation; external-unit-weight',
                weight_floor_fraction=request.get('weight_floor_fraction', 0.))


def realize_labels(config, model, mass, labels, seed):
    """Matched-terminal events of the drawn row labels, in draw order; row k uses subseed(seed, 'exclusive:' + k)."""
    catalog = {row.label: row for row in rows(model)}
    keys = {f'exclusive:{label}': (list(catalog[label].pdgs), catalog[label].expression) for label in set(labels)}
    counts = {key: 0 for key in keys}

    for label in labels:
        counts[f'exclusive:{label}'] += 1
    realized = realize_rows(config, mass, dict(sorted(keys.items())), counts, seed, 'matched')
    streams = {key: iter(events) for key, events in realized.items()}

    return [next(streams[f'exclusive:{label}']) for label in labels]


def sample_rows(config, request):
    """Events of request['rows'] = {label: count}, concatenated in that order.

    Each label's events depend only on (seed, label, count). An HNL label is a mixture of its flavour
    and charge-conjugate entries in hnl_rates, drawn in proportion to their widths at the squared mixings.
    """
    model, mass, seed = request['model'], float(request['mass']), request['seed']
    catalog = {row.label: row for row in rows(model)}
    entries = {}
    for label in request['rows']:
        if label not in catalog:
            raise ValueError(f'Unknown {model} decay-table row {label!r}')
        row = catalog[label]
        if row.partonic:
            raise ValueError(f'{model} row {label} is partonic (Jets); its final states come from generate or generate_all')
        if row.threshold >= mass:
            raise ValueError(f'{model} row {label} is kinematically closed at {mass:g} GeV')
        entries[label] = [(list(row.pdgs), row.expression, 1.)]
    if model == 'hnl' and request['rows']:
        channels = list(zip(*hnl_rates(mass, request['mixing'])))
        for label in request['rows']:
            entries[label] = [(ids, expression, p) for name, ids, expression, p in channels if name == label]
            if not sum(p for *_, p in entries[label]) > 0:
                raise ValueError(f'hnl row {label} has no width at {mass:g} GeV for these mixings')
    keys, counts, chosen = {}, {}, {}
    for label, n in request['rows'].items():
        weights = np.array([p for *_, p in entries[label]])
        chosen[label] = np.random.default_rng(subseed(seed, f'{label}:entries')).choice(len(weights), n, p=weights / weights.sum())
        for j, (ids, expression, _) in enumerate(entries[label]):
            if np.any(chosen[label] == j):
                keys[f'{label}:{j}'], counts[f'{label}:{j}'] = (ids, expression), int(np.sum(chosen[label] == j))
    realized = realize_rows(config, mass, keys, counts, seed, request.get('terminal', 'pythia'))
    events = []
    for label, entry in chosen.items():
        streams = {j: iter(realized[f'{label}:{j}']) for j in np.unique(entry)}
        events += [next(streams[j]) for j in entry]
    return events
