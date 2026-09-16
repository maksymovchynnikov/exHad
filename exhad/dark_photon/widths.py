"""Inclusive light-``uds`` width of the electromagnetic current on 1.70--5.00 GeV.

J_EM,uds = 2/3 u u - 1/3 d d - 1/3 s s with a unit coupling; multiply widths
by (epsilon e)^2.  Charm and bottom enter only through the running-alpha_s
flavour thresholds, never through this current.  Gamma_uds = f(m) R_uds(m)
Gamma_mumu(m), where f blends the DeLiVeR exclusive/perturbative ratio into 1
with a quintic smoothstep on 1.70--2.00 GeV and f = 1 above.
"""

from __future__ import annotations
import csv
import functools
import math
from .. import DATA

DOMAIN_MIN_GEV = 1.70

DOMAIN_MAX_GEV = 5.00

MUON_MASS_GEV = 0.1056583745

DELIVER_LAMBDA_QCD_GEV = 0.212

INCLUSIVE_MATCH_END_GEV = 2.00

BORN_R_UDS = 3.0 * math.fsum(charge * charge for charge in (2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0))


def _mass_in_domain(mass_gev: object) -> float:
    if isinstance(mass_gev, bool) or not isinstance(mass_gev, (int, float)):
        raise TypeError("dark-photon width mass must be a number")
    mass = float(mass_gev)
    if not math.isfinite(mass) or mass < DOMAIN_MIN_GEV or mass > DOMAIN_MAX_GEV:
        raise ValueError(f"dark-photon width mass {mass} lies outside [{DOMAIN_MIN_GEV}, {DOMAIN_MAX_GEV}] GeV")
    return mass


def deliver_alpha_s(mass_gev: object) -> float:
    """Four-loop alpha_s of DeLiVeR: Lambda_QCD = 0.212 GeV, n_f thresholds 1.27, 4.18, 172.76 GeV."""

    mass = _mass_in_domain(mass_gev)
    nf = 3.0 if mass < 1.27 else 4.0 if mass <= 4.18 else 5.0 if mass <= 172.76 else 6.0
    beta0 = 11.0 - (2.0 / 3.0) * nf
    beta1 = 102.0 - (38.0 / 3.0) * nf
    beta2 = (2857.0 / 2.0) - (5033.0 / 18.0) * nf + (325.0 / 54.0) * nf * nf
    beta3 = (100541.0 - 24423.3 * nf + 1625.4 * nf * nf - 27.493 * nf**3)
    log_scale = math.log(mass * mass / (DELIVER_LAMBDA_QCD_GEV**2))
    leading = 4.0 * math.pi / (beta0 * log_scale)
    log_log = math.log(log_scale)
    correction1 = -beta1 * log_log / (beta0**2 * log_scale)
    correction2 = (beta1**2 / (beta0**4 * log_scale**2)
                   * (log_log**2 - log_log - 1.0 + beta2 * beta0 / beta1**2))
    correction3 = (beta1**3 / (beta0**6 * log_scale**3)
                   * (-log_log**3 + 2.5 * log_log**2 + 2.0 * log_log - 0.5
                      - 3.0 * beta2 * beta0 * log_log / beta1**2
                      + beta3 * beta0**2 / (2.0 * beta1**3)))
    return leading * (1.0 + correction1 + correction2 + correction3)


def electromagnetic_uds_r(mass_gev: object) -> float:
    """R_uds = 3 sum Q^2 (1 + sum_n c_n (alpha_s/pi)^n), coefficients at n_f = 3 and zero singlet term."""

    mass = _mass_in_domain(mass_gev)
    coefficient_flavors = 3.0
    singlet_eta = 0.0
    coefficients = (
        1.0,
        1.9857 - 0.1152 * coefficient_flavors,
        -6.63694 - 1.20013 * coefficient_flavors - 0.00518 * coefficient_flavors**2 - 1.240 * singlet_eta,
        -156.61 + 18.775 * coefficient_flavors - 0.7974 * coefficient_flavors**2
        + 0.0215 * coefficient_flavors**3 - (17.828 - 0.575 * coefficient_flavors) * singlet_eta,
    )
    expansion = deliver_alpha_s(mass) / math.pi
    return BORN_R_UDS * (1.0 + math.fsum(
        coefficient * expansion**order for order, coefficient in enumerate(coefficients, start=1)))


def massive_muon_unit_current_width(mass_gev: object) -> float:
    """Gamma(V -> mu+ mu-) = m/(12 pi) (1+2x) sqrt(1-4x), x = (m_mu/m)^2, for unit coupling and charge."""

    mass = _mass_in_domain(mass_gev)
    ratio = (MUON_MASS_GEV / mass) ** 2
    return mass / (12.0 * math.pi) * (1.0 + 2.0 * ratio) * math.sqrt(1.0 - 4.0 * ratio)


@functools.lru_cache(maxsize=1)
def _inclusive_match_nodes() -> tuple[tuple[tuple[float, float], ...], tuple[float, ...]]:
    """Exclusive/perturbative ratio nodes on [1.70, 2.00] and their weighted-harmonic C1 slopes."""

    path = DATA / "dark-photon" / "exclusive_to_perturbative_ratio.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        nodes = tuple((float(row["m"]), float(row["exclusive_over_perturbative"]))
                      for row in csv.DictReader(stream)
                      if DOMAIN_MIN_GEV <= float(row["m"]) <= INCLUSIVE_MATCH_END_GEV)
    intervals = tuple(right[0] - left[0] for left, right in zip(nodes, nodes[1:]))
    secants = tuple((right[1] - left[1]) / interval for left, right, interval in zip(nodes, nodes[1:], intervals))
    slopes = [secants[0]]
    for index in range(1, len(nodes) - 1):
        left, right = secants[index - 1], secants[index]
        if left == 0.0 or right == 0.0 or left * right <= 0.0:
            slopes.append(0.0)
            continue
        weight_left = 2.0 * intervals[index] + intervals[index - 1]
        weight_right = intervals[index] + 2.0 * intervals[index - 1]
        slopes.append((weight_left + weight_right) / (weight_left / left + weight_right / right))
    slopes.append(secants[-1])
    return nodes, tuple(slopes)


def _exclusive_over_perturbative(mass: float) -> float:
    nodes, slopes = _inclusive_match_nodes()

    for index, (left, right) in enumerate(zip(nodes, nodes[1:])):
        if left[0] <= mass <= right[0]:
            width = right[0] - left[0]
            coordinate = (mass - left[0]) / width
            h00 = 2.0 * coordinate**3 - 3.0 * coordinate**2 + 1.0
            h10 = coordinate**3 - 2.0 * coordinate**2 + coordinate
            h01 = -2.0 * coordinate**3 + 3.0 * coordinate**2
            h11 = coordinate**3 - coordinate**2
            return h00 * left[1] + h10 * width * slopes[index] + h01 * right[1] + h11 * width * slopes[index + 1]
    raise RuntimeError("dark-photon inclusive matching mass is unsupported")


def unit_current_inclusive_uds_width(mass_gev: object) -> float:
    mass = _mass_in_domain(mass_gev)
    factor = 1.0
    if mass < INCLUSIVE_MATCH_END_GEV:
        coordinate = (mass - DOMAIN_MIN_GEV) / (INCLUSIVE_MATCH_END_GEV - DOMAIN_MIN_GEV)
        weight = 6.0 * coordinate**5 - 15.0 * coordinate**4 + 10.0 * coordinate**3
        factor = (1.0 - weight) * _exclusive_over_perturbative(mass) + weight
    return factor * electromagnetic_uds_r(mass) * massive_muon_unit_current_width(mass)
