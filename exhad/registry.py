"""Process-wide model generators, shared by the worker and the complete-decay palette."""
import atexit
from functools import lru_cache
import math
import os


@lru_cache(maxsize=None)
def dark_photon(variation):
    """Frozen light-uds model and its paired open-charm generator for one EM variation (None: central)."""
    from .dark_photon.model import build_light_model
    from .dark_photon.open_charm import build_pythia_charm_event_generator
    binary, xmldoc = (os.path.realpath(os.environ[name]) for name in ('EXHAD_RUNTIME_BINARY', 'EXHAD_PYTHIA8DATA'))
    light = build_light_model(binary=binary, xmldoc=xmldoc, em_variation_id=variation)
    atexit.register(light.close)
    charm = build_pythia_charm_event_generator(binary=binary, xmldoc=xmldoc)
    atexit.register(charm.close)

    return light, charm


def dark_photon_events(mass, count, seed, variation):
    """Six-field events of the disjoint light-uds plus open-charm EM measure; one light/charm draw per event."""

    if not math.isfinite(float(mass)) or not 1.70 <= float(mass) <= 5.00:
        raise ValueError('total-EM dark-photon mass must lie in [1.70,5.00] GeV')

    if not count:
        return []
    from .core.eventcalc import decay_rng, six_field
    from .dark_photon.open_charm import generate_total_em_dark_photon_events
    rng = decay_rng(seed)

    return [six_field(event, rng) for event in
            generate_total_em_dark_photon_events(*dark_photon(variation), float(mass), int(count), int(seed))]


@lru_cache(maxsize=None)
def b_l():
    """The b-l generator: light hadrons plus open charm."""
    from .b_l.open_charm import BaryonOpenCharmGenerator
    generator = BaryonOpenCharmGenerator()
    atexit.register(generator.close)
    return generator
