"""B-L hadronic decays below 2 GeV from the exclusive DeLiVeR rows.

data/b-l/eventcalc_branching_ratios.json holds partial widths per g_BL^2 (tools/build_bl_low_mass.py);
eventcalc_low_mass.json records the matching mass m_x, where the row sum last crosses the
perturbative width rates.total_width, and the kappa factors of the 2-GeV exclusive families.

* Rows above m_x: the 3pi and KKpi rows are scaled by 1 + S(m)(kappa - 1), S the quintic smoothstep
  from 0 at m_x to 1 at 2 GeV.
* m <= m_j: Gamma_had is the row sum; each event is one row. m_j (join_mass) is the last crossing of
  the interpolated, kappa-scaled row sum with Gamma_pert on [1.70, 1.80] GeV; it lies within 1e-4 GeV
  of m_x, which the table builder computed from direct DeLiVeR widths.
* m_j < m < 2 GeV: Gamma_had = rows + J(m)(Gamma_pert - rows) joins the row sum to the perturbative
  width, J the quintic smoothstep on [m_j, m_j + JOIN_GEV] (value, slope and curvature of each curve
  kept at both ends). The residual J(Gamma_pert - rows) belongs to the Model-1 remainder family of
  b-l: Pythia-pool events conditioned on that family, as at 2 GeV.

Events use the matched terminal convention; the kinematic seam at 2 GeV (flat rows below, Model-1
families above) is accepted and not smoothed.
"""
from functools import lru_cache
import json
import math
from types import SimpleNamespace
import numpy as np
from .. import DATA
from ..full_decays import interpolate, subseed
from . import rates as perturbative

JOIN_GEV = 0.02
# Pythia-pool share of remainder-family proposals of b-l below 2 GeV (2e3 proposals per source
# at 1.74, 1.75, 1.80, 1.90 GeV, source-weighted): >= 0.011. Half of it bounds the rejection budget.
REMAINDER_ACCEPTANCE_FLOOR = 0.005


@lru_cache(maxsize=1)
def _meta():
    return json.loads((DATA / 'b-l/eventcalc_low_mass.json').read_text())


def matching_mass():
    return float(_meta()['matching_mass_gev'])


def _smoothstep(x):
    x = min(1., max(0., x))
    return x * x * x * (10. - 15. * x + 6. * x * x)


def row_widths(mass):
    """Row partial widths per g_BL^2 at mass < 2 GeV, kappa-blended above m_x (threshold rule)."""
    from ..exclusive import rows, table_start
    catalog = rows('b-l')

    if not table_start('b-l') <= mass < 2.:
        raise ValueError(f"B-L exclusive widths cover {table_start('b-l'):g} <= mass < 2 GeV")
    widths = np.array([interpolate(row.values, mass, row.threshold) for row in catalog])
    m_x = matching_mass()

    if mass > m_x:
        s = _smoothstep((mass - m_x) / (2. - m_x))
        for kappa in (_meta()['kappa'][family] for family in ('three_pion', 'kkpi')):
            for i, row in enumerate(catalog):
                if row.label in kappa['rows']:
                    widths[i] *= 1. + s * (kappa['value'] - 1.)

    return widths


def _excess(mass):
    return perturbative.total_width(mass) - math.fsum(row_widths(mass))


@lru_cache(maxsize=1)
def join_mass():
    from scipy.optimize import brentq
    grid = np.arange(1.70, 1.80 + 5e-5, 1e-4)
    signs = np.sign([_excess(m) for m in grid])
    last = int(np.flatnonzero(signs[:-1] != signs[1:])[-1])
    if not signs[last] < 0 < signs[last + 1]:
        raise ValueError('B-L row sum has no last crossing with the perturbative width on [1.70, 1.80] GeV')
    return brentq(_excess, grid[last], grid[last + 1], xtol=1e-13, rtol=4 * np.finfo(float).eps)


def hadronic_widths(mass):
    """(row widths, remainder width) per g_BL^2; their sum is Gamma_had."""
    widths = row_widths(mass)
    m_j = join_mass()

    if mass <= m_j:
        return widths, 0.
    residual = perturbative.total_width(mass) - math.fsum(widths)

    if residual < 0.:
        raise ValueError(f'B-L exclusive rows exceed the perturbative width at {mass:g} GeV')

    return widths, _smoothstep((mass - m_j) / JOIN_GEV) * residual


def hadronic_width(mass):
    widths, remainder = hadronic_widths(mass)
    return math.fsum([*widths, remainder])


def sample(config, mass, count, seed):
    """Unweighted hadronic decays: a row (or the remainder owner) per event, in proportion to its width."""
    from ..exclusive import realize_labels, rows
    catalog = rows('b-l')
    widths, remainder = hadronic_widths(mass)
    labels = [row.label for row in catalog] + ['remainder']
    weights = np.array([*widths, remainder])

    if not weights.sum() > 0:
        raise ValueError(f'b-l has no open hadronic channel at {mass:g} GeV; use generate_all')
    chosen = np.random.default_rng(subseed(seed, 'exclusive-rows')).choice(len(weights), count, p=weights / weights.sum())
    drawn = [labels[j] for j in chosen]
    explicit = [label for label in drawn if label != 'remainder']
    events = iter(realize_labels(config, 'b-l', mass, explicit, seed))
    residual = iter(remainder_events(mass, len(drawn) - len(explicit), subseed(seed, 'exclusive:remainder')))

    return [next(residual) if label == 'remainder' else next(events) for label in drawn]


def remainder_events(mass, count, seed):
    """b-l Pythia-pool events conditioned on the remainder family, EventCalc six-field records.

    The prepared point accepts every remainder proposal (named families excluded, removed external
    modes vetoed as at 2 GeV); its budget is sized by REMAINDER_ACCEPTANCE_FLOOR, not the sealed card.
    The deployment card starts at 2 GeV, so the u/d/s source weights are held at their 2-GeV values;
    only the string mass follows the requested mass.
    """
    if count == 0:
        return []
    from contextlib import ExitStack
    from ..core.eventcalc import decay_rng, six_field
    from ..core.seeds import splitmix64
    from ..model1.sampler import _active_event, _fragmentation_runtime, _required_rejection_attempts, load_deployment
    deployment = load_deployment('b-l')
    model = deployment.family_model
    remainder, named = model.contract.remainder_family_id, frozenset(model.contract.named_family_ids)
    share = (1. - REMAINDER_ACCEPTANCE_FLOOR) / len(named)
    point = SimpleNamespace(
        mass_gev=mass, source_weights=model.evaluate(2.).source_weights,
        family_probabilities={**{family: 0. for family in named}, remainder: 1.},
        raw_family_probabilities={**{family: share for family in named}, remainder: REMAINDER_ACCEPTANCE_FLOOR},
        event_weights={**{family: 0. for family in named}, remainder: 1. / REMAINDER_ACCEPTANCE_FLOOR})
    budget = SimpleNamespace(**{**vars(deployment), 'maximum_attempts': _required_rejection_attempts(REMAINDER_ACCEPTANCE_FLOOR)})
    rng = decay_rng(seed)
    with ExitStack() as leases:
        runtime = leases.enter_context(_fragmentation_runtime(deployment, model))
        events = [_active_event(budget, model, runtime, mass, splitmix64(seed ^ splitmix64(index)), 'central',
                                excluded_families=named, prepared_point=point) for index in range(count)]
    return [[list(map(float, record[i:i + 6])) for i in range(0, len(record), 6)]
            for record in (six_field(event, rng) for event in events)]
