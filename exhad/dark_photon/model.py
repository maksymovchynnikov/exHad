"""Composition root of the dark-photon light-uds event model.

Binds the width inputs, the frozen EM response authority and the
DeLiVeR primary-mode authority to the reference and exclusive Pythia
runtimes.  Workers start lazily on the first generated event.
"""

from __future__ import annotations
import functools
from ..core.event_worker import ExclusiveInjectionRuntime, PythiaSubprocessRuntime, particle_catalog_from_pythia_xml
from .closure import load_inputs
from .em_response import EmResponseFit
from .measure import ConstructedEventMeasure
from .modes import DarkPhotonModeAuthority

REFERENCE_PROVIDER_ID = "pythia317-dark-photon-reference"


def build_light_model(*, binary, xmldoc, em_variation_id=None) -> ConstructedEventMeasure:
    """Build the production model for one EM covariance variation (None: central)."""

    inputs = load_inputs()
    authority = EmResponseFit(em_variation_id)
    modes = DarkPhotonModeAuthority()
    edge = inputs.scales.fragmentation_edge_gev
    # The high reference cap is required by exact rejection sampling of the
    # small unowned complement at 1.70 GeV and bounds when the flat
    # phase-space pion fallback starts; it changes no accepted distribution.
    reference = PythiaSubprocessRuntime(binary, xmldoc, maximum_condition_attempts=65_536, maximum_workers=3)
    # Rare retained prompt responses (omega -> pi+ pi-) need a large cap to
    # force the already selected mode onto its rate-owned channel.
    exclusive = ExclusiveInjectionRuntime(
        binary, xmldoc,
        {realizer_id: functools.partial(modes.modes_for, realizer_id) for realizer_id in modes.realizer_ids},
        maximum_condition_attempts=1024,
        maximum_workers=len({mode.daughter_pdgs for realizer_id in modes.realizer_ids
                             for mode in modes.modes_for(realizer_id, edge)}),
        injected_mass_mode="width")
    (exclusive_provider_id,) = set(inputs.exclusive_edge.providers.values())
    providers = {exclusive_provider_id: exclusive.generate, REFERENCE_PROVIDER_ID: reference.generate}
    # A fitted response realizes only the modes its realizer has at the edge.
    for route in authority.routes:
        providers[route.provider_id] = functools.partial(
            exclusive.generate, allowed_mode_ids=frozenset(
                mode.mode_id for mode in exclusive.modes_at(route.realizer_id, edge)))
    return ConstructedEventMeasure(
        inputs, authority, particle_catalog_from_pythia_xml(xmldoc), providers,
        REFERENCE_PROVIDER_ID, (exclusive, reference))
