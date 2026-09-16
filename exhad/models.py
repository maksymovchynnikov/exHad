"""Public model names and request checks."""
import json
import math
from typing import NamedTuple

MODELS = ('dark-photon', 'alp-fermion', 'scalar-central', 'scalar-lower',
          'scalar-upper', 'scalar-1809', 'b-l', 'hnl')
SCALARS = {'scalar-central': '2407.13587_central', 'scalar-lower': '2407.13587_lower',
           'scalar-upper': '2407.13587_upper', 'scalar-1809': '1809.01876'}


class SourceQuantumNumbers(NamedTuple):
    """The decaying particle as the hadronizer takes it: 2J, P, C (None when the source
    is not a C eigenstate) and the electric charge of the source current."""

    spin_doubled: int
    parity: int
    c_parity: int | None
    charge: int

    @property
    def arguments(self) -> tuple[str, ...]:
        """Its four protocol tokens, in the order the C++ side reads them."""
        return (str(self.spin_doubled), str(self.parity),
                'none' if self.c_parity is None else str(self.c_parity), str(self.charge))


# The quantum numbers of every source exHad hadronizes, written down once.  Whether a
# spin-1 source is the parent's spin or an average over its polarizations is NOT one of
# them: the boson portals carry it, the heavy-neutral-lepton currents average it, and
# that fact travels beside the quantum numbers under its own name.
VECTOR_SOURCE = SourceQuantumNumbers(2, -1, -1, 0)
PSEUDOSCALAR_SOURCE = SourceQuantumNumbers(0, -1, 1, 0)
SCALAR_SOURCE = SourceQuantumNumbers(0, 1, 1, 0)
CHARGED_CURRENT_SOURCE = SourceQuantumNumbers(2, -1, None, -1)
NEUTRAL_CURRENT_SOURCE = SourceQuantumNumbers(2, -1, None, 0)
PARENT_SPIN_WORD = {True: 'carried', False: 'averaged'}
# Each boson portal's decaying particle; an HNL's is the current its q-qbar pair came from.
MODEL_SOURCE = {'dark-photon': VECTOR_SOURCE, 'b-l': VECTOR_SOURCE, 'alp-fermion': PSEUDOSCALAR_SOURCE,
                **dict.fromkeys(SCALARS, SCALAR_SOURCE)}
# Parent masses every hnl interface accepts: exHad covers GeV-scale masses, and the tables
# below this edge are not used.
HNL_MASS_GEV = (0.02, 40.)
# Matched supports of the models without a Model-1 deployment. HNL: parent masses
# whose every q-qbar W lies in the exact-W routes (generation continues to HNL_MASS_GEV[1]).
SUPPORT_GEV = {'dark-photon': [1.70, 5.00], 'hnl': [HNL_MASS_GEV[0], 5.27]}
# PDG identifier of each model's decaying particle: the hidden-U(1) gamma_v (4900022), Z' (32),
# H0 (35), A0 (36) and Pythia's Majorana nu_Re (9900012, self-conjugate).
MOTHER_PDG = {'dark-photon': 4900022, 'alp-fermion': 36, 'b-l': 32, 'hnl': 9900012,
              **dict.fromkeys(SCALARS, 35)}


def decays_with_hadrons(model):
    """Name of the decays that contain hadrons: a heavy neutral lepton decays semileptonically
    (a charged lepton or a neutrino together with hadrons), a boson portal hadronically."""
    return 'semileptonic' if validate_model(str(model)) == 'hnl' else 'hadronic'


def validate_model(model):
    if model not in MODELS:
        raise ValueError(f'Unknown model {model!r}; choose from: {", ".join(MODELS)}')
    return model


def mother_pdg(model, override=None):
    """The model's MOTHER_PDG, or override, a nonzero signed 32-bit PDG identifier of the host."""
    if override is None:
        return MOTHER_PDG[validate_model(str(model))]
    if isinstance(override, bool) or not isinstance(override, int) or override == 0 or abs(override) >= 2**31:
        raise ValueError('mother PDG identifier must be a nonzero signed 32-bit integer')
    return override


def eventcalc_tables(model):
    """Paths below DATA of the EventCalc-format rate inputs of a public model.

    B-L has no EventCalc model: its decay table holds DeLiVeR partial widths per g_BL^2 below 2 GeV
    (tools/build_bl_low_mass.py) and low_mass the matching mass and kappa factors.
    """
    if model == 'b-l':
        return {'decay': 'b-l/eventcalc_branching_ratios.json', 'low_mass': 'b-l/eventcalc_low_mass.json'}
    if model in SCALARS:
        return {'decay': f'scalar/eventcalc_branching_ratios_{SCALARS[model]}.json',
                'ctau': f'scalar/eventcalc_ctau_{SCALARS[model]}.json'}
    return {'dark-photon': {'decay': 'dark-photon/eventcalc_branching_ratios.json',
                            'ctau': 'dark-photon/eventcalc_ctau.txt'},
            'alp-fermion': {'decay': 'alp-fermion/eventcalc_branching_ratios.json',
                            'ctau': 'alp-fermion/eventcalc_ctau.txt'},
            'hnl': {'decay': 'hnl/eventcalc_branching_ratios.json',
                    'widths': 'hnl/eventcalc_total_widths.dat'}}.get(model, {})


def check_numbers(mass, events, seed, check_mass=True):
    if isinstance(events, bool) or not isinstance(events, int) or events < 0:
        raise ValueError('events must be a nonnegative integer')
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError('seed must be an unsigned 64-bit integer')
    if not math.isfinite(float(mass)) or float(mass) <= 0:
        raise ValueError('mass must be finite and positive')


def check_request(model, method, mass, events, seed, *, weighted=False,
                  weight_floor_fraction=0., mixing=None, check_mass=True, rows=None, terminal='pythia'):
    """Argument checks of the public methods of Generator.

    method is the public method name; all failures raise ValueError. Physics
    domains (mass windows, variations, HNL primaries) stay with the worker.
    """
    if method == 'hadronize_hnl' and model != 'hnl':
        raise ValueError('hadronize_hnl requires Generator("hnl")')
    if (weighted or method == 'generate_weighted') and model != 'alp-fermion':
        raise ValueError('Weighted sampling supports only alp-fermion')
    if not isinstance(weighted, bool):
        raise ValueError('weighted must be a boolean')
    if method in ('generate_all', 'generate_weighted') and (
            isinstance(weight_floor_fraction, bool)
            or not isinstance(weight_floor_fraction, (int, float))
            or not math.isfinite(weight_floor_fraction) or not 0 <= weight_floor_fraction <= 1):
        raise ValueError('weight_floor_fraction must lie in [0, 1]')
    if method == 'generate_rows' and (
            not isinstance(rows, dict) or not all(isinstance(label, str) and label for label in rows)
            or terminal not in ('pythia', 'matched')):
        raise ValueError("rows must map decay-table row labels to event counts; terminal is 'pythia' or 'matched'")
    for count in (rows or {}).values():
        check_numbers(mass, count, seed, check_mass)
    if method in ('generate', 'generate_all', 'generate_rows') and model == 'hnl':
        if (mixing is None or len(mixing) != 3
                or any(isinstance(x, bool) or not math.isfinite(x) or x < 0 for x in mixing)
                or sum(mixing) <= 0):
            raise ValueError('HNL requires three nonnegative squared mixings with a positive sum')
    elif method in ('generate', 'generate_all', 'generate_rows') and mixing is not None:
        raise ValueError('mixing is only applicable to HNL')
    # HNL primaries are counted with len(), so their count needs no check.
    check_numbers(mass, 0 if method == 'hadronize_hnl' else events, seed, check_mass)


def model_info(model):
    """Plain data a simulation host needs to own the matched rows; starts no worker.

    mother_pdg is the default PDG identifier of the decaying particle (MOTHER_PDG).
    support_gev is the mass interval of the matched generator; for the scalars it
    extends to the showered 63-GeV end of generation. generation_gev holds the mass
    intervals of generate (decays_with_hadrons(model), open above the lowest threshold of a row
    with hadrons) and generate_all ('all', from the decay-table start). Below support_gev[0] hadronic decays
    are exclusive table rows: exclusive_gev is that interval and exclusive_rows the rows
    (label and sorted PDG signature). tables holds the
    absolute paths of the EventCalc rate inputs the model is matched to. For the
    alp-fermion and scalar Model-1 deployments, eventcalc_rows maps each sorted decay-table
    PDG signature to its outer-owner row, and owner_probabilities holds that owner's
    mass nodes and per-row probabilities (piecewise linear, normalized at the mass).
    Scalar owners are the matched rows up to 4 GeV, the supplied hadronic rows from
    5 GeV, and their transition mixture between (exhad.transition).
    """
    from . import DATA
    model = validate_model(str(model))
    tables = eventcalc_tables(model)
    info = dict(model=model, mother_pdg=MOTHER_PDG[model], support_gev=SUPPORT_GEV.get(model),
                tables={key: str(DATA / path) for key, path in tables.items()})
    deployment = json.loads((DATA / 'common/model1_deployments.json').read_text())[
        'deployments'].get(model)
    if deployment is not None:
        card = json.loads((DATA / deployment['path']).read_text())
        info['support_gev'] = card['activation_support_gev']
        if 'low_mass' not in tables:
            owners = json.loads((DATA / card['outer_owner_authority']).read_text())
            if model in SCALARS:
                owners, info['support_gev'] = scalar_owners(owners, DATA / tables['decay']), [2.0, 63.0]
            info['eventcalc_rows'] = [
                {'pdg_signature': sorted(int(p) for p in row[1] if int(p) != -999), 'authority_row_id': row[0]}
                for row in json.loads((DATA / tables['decay']).read_text()) if row[0] in owners['probabilities']]
            info['owner_probabilities'] = owners
    from . import exclusive
    if model == 'hnl':
        info['generation_gev'] = {decays_with_hadrons(model): [exclusive.hadron_row_threshold(model), HNL_MASS_GEV[1]],
                                  'all': [HNL_MASS_GEV[0], HNL_MASS_GEV[1]]}
    else:
        high = info['support_gev'][1]
        info['generation_gev'] = {decays_with_hadrons(model): [exclusive.hadron_row_threshold(model), high],
                                  'all': [exclusive.table_start(model), high]}
        info['exclusive_gev'] = [exclusive.table_start(model), exclusive.START[model]]
        info['exclusive_rows'] = [{'label': row.label, 'pdg_signature': sorted(row.pdgs)}
                                  for row in exclusive.rows(model) if row.has_hadrons and not row.partonic]
    return info


def scalar_owners(owners, decay):
    """Row weights of the complete scalar sample: (1-h) matched owners + h supplied rows.

    Both sum to the hadronic branching fraction at each mass, so width, charm, bottom
    and nucleon rates are kept; the two-meson continuation vanishes with 1-h.
    """
    import numpy as np
    from .core.pdg import is_lepton
    from .transition import pythia_fraction
    low = np.asarray(owners['masses'])
    table = json.loads(decay.read_text())
    grid = np.asarray(table[0][2])[:, 0]
    masses = np.concatenate([low, grid[grid > low[-1]]])
    h, probabilities = pythia_fraction(masses), {}
    for label, pdgs, values in (row[:3] for row in table):
        values = np.asarray(values, dtype=float)
        hadronic = not all(is_lepton(p) or int(p) in (22, -999) for p in pdgs)
        if label in owners['probabilities'] or hadronic and np.any(values[values[:, 0] >= 4., 1] > 0):
            matched = np.zeros(len(masses))
            matched[:len(low)] = owners['probabilities'].get(label, 0.)
            probabilities[label] = ((1 - h) * matched + h * np.interp(masses, values[:, 0], values[:, 1])).tolist()
    return dict(owners, masses=masses.tolist(), probabilities=probabilities)
