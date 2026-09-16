"""Pseudoscalar event response for the alp-fermion's separately normalized Jets-cc row.

The row width comes from the alp-fermion calculation. The events split it between an
on-shell open-charm response, activated as p^3 of the neutral D D* pair and
reaching one at the 4 GeV handoff, and a light gluon response for the
complement. Both use the alp-fermion 0-+, I=0 filter on primary hadrons; the branch is
drawn once, before generation retries. Off-shell D D pi / D D gamma modes are
not resolved separately.
"""

from collections import Counter
import functools
import math
from pathlib import Path
import random
from ..core.event_model import ConservedCharges, EventModelError, validate_complete_event
from ..core.event_worker import EventWorker, PythiaSubprocessError, WorkerPool, particle_catalog_from_pythia_xml, retry, worker_settings
from ..core.eventcalc import six_field
from ..core.ownership import OwnershipCutError, derive_ownership_cut
from ..core.seeds import splitmix64
from ..models import PSEUDOSCALAR_SOURCE

HANDOFF_GEV = 4.0

_BRANCH_TAG = 0xAC125B31
_ATTEMPT_TAG = 0xAC125B32
_DECAY_TAG = 0xAC125B33
_ISOSPIN_TAG = 0xAC125B34


def _is_open_charm(pdg_id: int) -> bool:
    absolute = abs(int(pdg_id))
    return absolute >= 100 and ((absolute // 100) % 10 == 4 or (absolute // 1000) % 10 == 4)


def _check_charm_primary(roots) -> None:
    """Raise unless the primary roots carry closed open charm other than a 0-+ forbidden D Dbar pair."""
    roots = tuple(int(value) for value in roots)
    charm = tuple(value for value in roots if _is_open_charm(value))

    if not roots or not charm:
        raise PythiaSubprocessError("alp-fermion charm response has no open-charm primary hadron")

    if sum(1 if value > 0 else -1 for value in charm) != 0:
        raise PythiaSubprocessError("alp-fermion charm primary roots do not close charm flavor")

    if len(roots) == 2 and all(abs(value) in (411, 421, 431) for value in roots):
        raise PythiaSubprocessError("0-+ alp-fermion response produced the forbidden D Dbar two-pseudoscalar primary state")


def open_charm_fraction(mass, d_mass, dstar_mass):
    """Open-charm share of the nominal charm width, with the generator's pole masses."""
    threshold = d_mass + dstar_mass

    if mass <= threshold:
        return 0.0

    if mass >= HANDOFF_GEV:
        return 1.0

    def momentum_squared(m):
        return ((m * m - threshold * threshold) * (m * m - (d_mass - dstar_mass)**2) / (4 * m * m))

    return (momentum_squared(mass) / momentum_squared(HANDOFF_GEV))**1.5


class AlpFermionCharmRuntime:
    def __init__(self, binary, xmldoc, settings=(), stop_mass=0.):
        self.pool = WorkerPool(8)
        self.binary, self.xmldoc = binary, xmldoc
        self.settings = worker_settings(settings)
        self.stop_mass = float(stop_mass)
        self.catalog = particle_catalog_from_pythia_xml(xmldoc)
        self.d_mass = self.catalog[421].mass_gev
        self.dstar_mass = self.catalog[423].mass_gev

    def close(self):
        self.pool.close()

    def component_for(self, mass, seed):
        unit = (splitmix64(seed ^ _BRANCH_TAG) >> 11) / float(1 << 53)
        return "charm" if unit < open_charm_fraction(float(mass), self.d_mass, self.dstar_mass) else "glue"

    def generate_component(self, mass, seed, component):
        """One filtered event of the chosen component; retries never switch component."""
        mass = float(mass)
        if not math.isfinite(mass) or not 3.5 <= mass <= 5.0:
            raise ValueError("alp-fermion charm-row response support is 3.5--5 GeV")

        def accept(event):
            primary = event.primary_pdgs

            if component == "charm":
                _check_charm_primary(primary)
            elif any(_is_open_charm(pdg) for pdg in primary):
                return False  # the glue branch represents light hadrons only

            try:
                validate_complete_event(event, parent_mass_gev=mass,
                                        target_charges=ConservedCharges(), catalog=self.catalog)
            except EventModelError as error:
                # Retry rare numerical closure failures in secondary decays;
                # malformed graphs and charges still fail.
                if (str(error).startswith('event does not conserve ')
                        or str(error).endswith('is outside the mass shell tolerance')):
                    return False
                raise

            return True

        with self.pool.use(component, lambda: EventWorker(
                self.binary, self.xmldoc, component, PSEUDOSCALAR_SOURCE, True,
                "qq" if component == "charm" else "gg",
                stop_mass_gev=self.stop_mass, settings=self.settings)) as worker:
            event = retry((splitmix64(seed ^ _ATTEMPT_TAG ^ splitmix64(attempt)) for attempt in range(256)),
                          functools.partial(worker.request, mass), accept)
            if event is None:
                raise RuntimeError("alp-fermion response exhausted attempts without changing its selected branch")
            return event

    def generate(self, mass, seed):
        """Branch event; eta(')pipi in the glue branch is projected to the I=0 2:1 charge ratio.

        The eta subgroup is kept and its charge state drawn (charged with 2/3);
        events are rejected until that exact state, which keeps Pythia's
        momentum law conditional on it and the subgroup rate unchanged.
        """
        component = self.component_for(mass, seed)
        event = self.generate_component(mass, seed, component)
        if component != 'glue':
            return event

        def state(candidate):
            try:
                return Counter(derive_ownership_cut(candidate).pdg_ids)
            except OwnershipCutError:
                return Counter()
        initial = state(event)
        eta = next((pdg for pdg in (221, 331) if initial in (
            Counter((pdg, 211, -211)), Counter((pdg, 111, 111)))), None)
        if eta is None:
            return event
        charged = (splitmix64(seed ^ _ISOSPIN_TAG) >> 11) / float(1 << 53) < 2 / 3
        target = Counter((eta, 211, -211) if charged else (eta, 111, 111))
        if initial == target:
            return event
        for attempt in range(1, 1 << 20):
            candidate = self.generate_component(mass, splitmix64(seed ^ _ISOSPIN_TAG ^ splitmix64(attempt)), 'glue')
            if state(candidate) == target:
                return candidate
        raise RuntimeError('alp-fermion light dipion charge conditioning exhausted its fixed target')


def realize_alp_fermion_charm_row(*, mass_gev, count, seed, row_id, configuration):
    """Realize the selected Jets-cc row without changing its supplied width."""
    from .realizers import RUNTIMES
    binary = Path(configuration["runtime_binary"]).resolve()
    xmldoc = Path(configuration["runtime_xmldoc"]).resolve()
    settings = tuple(configuration["pythia_settings"])
    stop_mass = float(configuration["stop_mass_gev"])
    key = ("alp-fermion-charm", binary, xmldoc, settings, stop_mass)

    with RUNTIMES.use(key, lambda: AlpFermionCharmRuntime(binary, xmldoc, settings, stop_mass)) as runtime:
        result = []
        for i in range(count):
            word = seed if count == 1 else splitmix64(seed ^ splitmix64(i))
            result.append(six_field(runtime.generate(mass_gev, word), random.Random(splitmix64(word ^ _DECAY_TAG)), fsum=True))
        return result
