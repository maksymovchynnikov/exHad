"""Light-hadron rates of the B-L (baryon-number) vector current on 2--5 GeV.

J_B = (ubar u + dbar d + sbar s)/3.  All widths are per unit g_B^2 in GeV.

- External owners: native DeLiVeR form factors with q_u = q_d = q_s = 1/3
  (pi0 gamma, eta gamma, K Kbar charged + neutral, p pbar, n nbar).  The
  form-factor modules keep their couplings in module globals shared with the
  dark-photon code, so every native evaluation resets them first.
- Inclusive light width: perturbative B-current R ratio times m/(12 pi).
- Active (fragmentation) width: inclusive minus externals.
- Six named active families: exclusive widths on [2, 3] GeV (EM-fit three
  pion times the native B/EM ratio, native KKpi splines, omega/phi parent
  modes folded through their daughter BRs), W_f(3) (3/m)^{2p} x active and
  Pythia-matching ratios above 3 GeV, and a quintic blend on [2, 3] GeV.
"""

from __future__ import annotations
import functools
import importlib
import json
import math
from pathlib import Path
import sys
import numpy as np
from .. import DATA, ROOT
from ..dark_photon.widths import deliver_alpha_s

# Fixed omega/phi decay branchings (PDG) of the isoscalar exclusive projection.
BR_OMEGA_3PI = 0.892

BR_PHI_CHARGED_KK = 0.491

BR_PHI_NEUTRAL_KK = 0.340

BR_PHI_3PI = 0.154

NAMED_FAMILIES = (
    "exact-3pi", "exact-kkpi", "exact-5pi", "eta-three-pion",
    "eta-kaon-pair", "exact-kkpipi",
)

FAMILIES = (*NAMED_FAMILIES, "remainder")

_CHARGES = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

ALPHA_ZERO = 1 / 137.035999084

GEV2_TO_NB = 389379.3656


class BaryonRateError(ValueError):
    """A B-L rate guard failed."""


@functools.cache
def _inputs():
    """Exclusive-width splines and scales of data/b-l/rates.json (PPoly: no extrapolation, axis 0)."""
    from scipy.interpolate import PPoly
    data = json.loads((DATA / "b-l" / "rates.json").read_text())
    for key in ("three_pion_splines", "isoscalar_splines", "kkpi_splines"):
        data[key] = {name: PPoly.construct_fast(np.array(item["c"]), np.array(item["x"]), False)
                     for name, item in data[key].items()}
    return data


def _form_factor(name: str):
    """Return a DeLiVeR form-factor module reset to the B-current charges."""
    deliver = (ROOT / "external" / "deliver").resolve()

    if str(deliver) not in sys.path:
        sys.path.insert(0, str(deliver))
    module = importlib.import_module("src.form_factors." + name)
    expected = deliver / "src" / "form_factors" / (name + ".py")

    if Path(module.__file__).resolve() != expected.resolve():
        raise BaryonRateError(
            f"native form factor {name} resolved to {module.__file__}, not {expected}")
    module.resetParameters(1.0, 0.0, 0.0, 0.0, *_CHARGES)

    return module


def _nonnegative(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise BaryonRateError(
            f"native {label} width is not finite and nonnegative")
    return value


def kaon_pair_widths(mass: float) -> tuple[float, float]:
    """Native (charged, neutral) two-kaon widths of the B current."""
    kaon = _form_factor("FK")
    return (_nonnegative(kaon.GammaDM_mode(mass, 1), "charged-kaon"),
            _nonnegative(kaon.GammaDM_mode(mass, 0), "neutral-kaon"))


@functools.lru_cache(maxsize=4096)
def external_widths(mass: float) -> dict[str, float]:
    """External-owner widths in the owner-draw order (do not mutate)."""

    return {
        "pi0-gamma-block": _nonnegative(_form_factor("FPiGamma").GammaDM(mass), "pi0-gamma"),
        "eta-gamma-block": _nonnegative(_form_factor("FEtaGamma").GammaDM(mass), "eta-gamma"),
        "kaon-pair-block": math.fsum(kaon_pair_widths(mass)),
        "proton-pair-block": _nonnegative(_form_factor("Fppbar").GammaPDM(mass), "proton-pair"),
        "neutron-pair-block": _nonnegative(_form_factor("Fppbar").GammaNDM(mass), "neutron-pair"),
    }


def total_width(mass: float) -> float:
    """DeLiVeR's light-current R ratio for q_u=q_d=q_s=1/3 (n_f=3, eta=1).

    Born R = 1; the prefactor is the massless unit-current width m/(12 pi).
    """
    a = deliver_alpha_s(float(mass)) / math.pi
    nf, eta = 3., 1.
    coefficients = (1., 1.9857 - .1152 * nf,
                    -6.63694 - 1.20013 * nf - .00518 * nf**2 - 1.240 * eta,
                    -156.61 + 18.775 * nf - .7974 * nf**2 + .0215 * nf**3
                    - (17.828 - .575 * nf) * eta)
    return float(mass) / (12 * math.pi) * (
        1. + math.fsum(c * a**n for n, c in enumerate(coefficients, 1)))


def active_width(mass: float) -> float:
    active = total_width(mass) - math.fsum(external_widths(mass).values())
    if active <= 0 or not math.isfinite(active):
        raise BaryonRateError("inclusive width leaves no fragmentation width")
    return active


def smooth_weight(mass: float, interval=(2.0, 3.0)) -> float:
    z = min(1.0, max(0.0, (mass - interval[0]) / (interval[1] - interval[0])))
    return z**3 * (10.0 - 15.0 * z + 6.0 * z * z)


def _three_pion_width(mass: float) -> float:
    """EM-fit sigma times the native B/EM ratio, joined on 2.0--2.2 GeV to the native B spline."""
    data = _inputs()
    spline = data["three_pion_splines"]
    native = data["three_pion_scale"] * float(mass * spline["bminusl"](mass))

    if mass >= 2.2:
        return native
    em, bl = float(mass * spline["em"](mass)), float(mass * spline["bminusl"](mass))

    if em <= 0 or bl <= 0:
        raise BaryonRateError("native current contraction must be positive")
    x = np.log(np.asarray(mass) / 2.)
    sigma = float(np.exp(sum(c * x**power for power, c in enumerate(data["three_pion_coefficients"]))))
    sigma_mu = 4 * math.pi * ALPHA_ZERO**2 / (3 * mass**2) * GEV2_TO_NB
    fit = mass / (12 * math.pi) * sigma / sigma_mu * bl / em

    if fit <= 0:
        raise BaryonRateError("corrected three-pion width is invalid")
    h = smooth_weight(mass, [2.0, 2.2])

    return (1 - h) * fit + h * native


@functools.lru_cache(maxsize=4096)
def known_widths(mass: float) -> dict[str, float]:
    """Named-family exclusive widths on [2, 3] GeV (do not mutate)."""
    data = _inputs()
    modes = {}

    for mode, spline in data["isoscalar_splines"].items():
        modes[mode] = 0.0 if mass <= data["isoscalar_thresholds"][mode] else float(spline(mass))
        if modes[mode] < 0 or not math.isfinite(modes[mode]):
            raise BaryonRateError(f"invalid isoscalar spline width for {mode} at {mass}")
    omega_pions = math.fsum((modes["omega-pi0-pi0"], modes["omega-piplus-piminus"]))
    phi_pions = math.fsum((modes["phi-pi0-pi0"], modes["phi-piplus-piminus"]))
    eta_omega, eta_phi = modes["eta-omega"], modes["eta-phi"]
    kkpipi = math.fsum(modes[f"kkpipi-kstar-band-{band}"] for band in range(4))
    phi_kk = BR_PHI_CHARGED_KK + BR_PHI_NEUTRAL_KK
    kkpi = [float(spline(mass)) for spline in data["kkpi_splines"].values()]

    if any(not math.isfinite(value) or value < -1e-14 for value in kkpi):
        raise BaryonRateError("invalid native KKpi interpolation")

    return {
        "exact-3pi": _three_pion_width(mass),
        "exact-kkpi": data["kkpi_scale"] * (math.fsum(max(0.0, value) for value in kkpi) / 3.0),
        "exact-5pi": math.fsum((BR_OMEGA_3PI * omega_pions, BR_PHI_3PI * phi_pions)),
        "eta-three-pion": math.fsum((BR_OMEGA_3PI * eta_omega, BR_PHI_3PI * eta_phi)),
        "eta-kaon-pair": phi_kk * eta_phi,
        "exact-kkpipi": math.fsum((1.0 * kkpipi, phi_kk * phi_pions)),
    }


def _high_widths(mass: float, model, power: float, saturation_off: bool) -> dict[str, float]:
    """W_f(3) (3/m)^{2p} x active(m)/active(3) x matching_f(m)/matching_f(3)."""
    edge = known_widths(3.0)
    high_active = active_width(mass)
    scale = (3.0 / mass)**(2 * power) * high_active / active_width(3.0)
    widths = {family: edge[family] * scale * (
        1.0 if saturation_off else model._matching_at(family, mass) / model._matching_at(family, 3.0))
        for family in NAMED_FAMILIES}

    if (high_active - math.fsum(widths.values()) < -2e-12
            or any(not math.isfinite(w) or w < 0 for w in widths.values())):
        raise BaryonRateError("exclusive-anchored high-mass widths exceed the inclusive target")

    return widths


def family_probabilities(mass: float, model, power: float, saturation_off: bool) -> dict[str, float]:
    """Conditional active-family probabilities, remainder absorbing rounding."""
    mass = float(mass)

    if mass <= 2.:
        widths = dict(known_widths(mass))
    else:
        high = _high_widths(mass, model, float(power), saturation_off)
        if mass >= 3.:
            widths = high
        else:
            h = smooth_weight(mass)
            known = known_widths(mass)
            widths = {f: (1 - h) * known[f] + h * high[f] for f in NAMED_FAMILIES}
    active = active_width(mass)
    remainder = active - math.fsum(widths.values())

    if remainder < -2e-12:
        raise BaryonRateError("named channel widths exceed the inclusive total")
    widths["remainder"] = max(0., remainder)
    probabilities = {f: widths[f] / active for f in FAMILIES}
    probabilities["remainder"] += 1. - math.fsum(probabilities.values())

    if min(probabilities.values()) < 0:
        raise BaryonRateError("family probabilities do not close positively")

    return probabilities
