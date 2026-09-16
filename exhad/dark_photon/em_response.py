"""Frozen electromagnetic spectral fit as the post-edge rate authority.

Seven coherent responses (3pi, 4pi charged/neutral, 6pi charged/neutral,
KKpipi charged/neutral) become fractions of the light-uds inclusive width:
the frozen edge anchors at exactly 2.0 GeV, a cubic Bernstein seam with
per-variation controls on (2.0, 2.5) GeV, and |sum_k B_k(s) c_k|^2 / R_uds from
2.5 GeV.  Each amplitude has constant-width poles M^2/(M^2 - s - i M Gamma)
and one conformal continuum z^k (1-z)^(2p) on the upper rim s + i0.  The
residual Gamma_uds - Gamma_external - sum(Gamma_response) must not be
materially negative for the central fit or any of the 62 eigenmode
variations.
"""

from __future__ import annotations
import json
import math
from types import SimpleNamespace
import numpy as np
from .. import DATA
from ..core.width_inputs import floating_tolerance
from .measure import PostEdgeRatePoint
from .widths import electromagnetic_uds_r

RESPONSE_SEAM_JOIN_GEV = 2.50

SIGMA_MU_BORN_NB_GEV2 = 86.8544769  # 4 pi alpha(0)^2 / 3; cancels in the fractions


class EmResponseError(ValueError):
    """The EM response fit yields no valid rate point at this mass."""


def _pole(s: np.ndarray, mass_gev: float, width_gev: float) -> np.ndarray:
    """Normalized constant-width pole ``M^2/(M^2-s-i*M*Gamma)``."""
    mass2 = mass_gev * mass_gev
    return mass2 / (mass2 - s - 1j * mass_gev * np.full(s.shape, width_gev, dtype=np.float64))


def _continuum(s: np.ndarray, cut_gev2: float, expansion_point_gev2: float, order: int,
               outer_falloff_power: int) -> np.ndarray:
    """Continuum ``z^k (1-z)^(2p)`` with ``sqrt(cut-s-i0) = -i sqrt(s-cut)``."""
    reference = math.sqrt(cut_gev2 - expansion_point_gev2)
    root = (np.sqrt(np.maximum(cut_gev2 - s, 0.0)).astype(np.complex128)
            - 1j * np.sqrt(np.maximum(s - cut_gev2, 0.0)))
    z = (root - reference) / (root + reference)
    one_minus_z = 2.0 * reference / (root + reference)  # 1-z without cancellation
    return np.power(z, order) * np.power(one_minus_z, 2 * outer_falloff_power)


class EmResponseFit:
    """One fit variation of data/dark-photon/em_response.json bound to its dark-photon response routes.

    Each response block is its ordered poles, then one continuum, with the
    variation's complex coefficients; the central fit is stored as ``None``.
    """

    def __init__(self, variation_id: str | None = None):
        data = json.loads((DATA / "dark-photon" / "em_response.json").read_text())

        if str(variation_id) not in data["variations"]:
            raise ValueError("Unknown dark-photon EM variation: " + str(variation_id))
        variation = data["variations"][str(variation_id)]
        self.support_gev = tuple(data["support_gev"])
        self.edge_fraction_anchors = data["edge_fraction_anchors"]
        self.routes = tuple(SimpleNamespace(**route) for route in data["routes"])
        self.blocks = {key: (block, np.array([complex(*value) for value in variation["coefficients"][key]]))
                       for key, block in data["blocks"].items()}
        self.seam_simplex_controls = variation["seam_simplex_controls"]

    def response_fractions_at(self, mass: float) -> dict[str, float]:
        if mass >= RESPONSE_SEAM_JOIN_GEV:
            s = np.asarray([mass * mass], dtype=np.float64)
            flux = SIGMA_MU_BORN_NB_GEV2 / (mass * mass)
            normalization = electromagnetic_uds_r(mass) * flux
            values = {}
            for route in self.routes:
                block, local = self.blocks[route.response_block_id]
                terms = [_pole(s, **pole) for pole in block["poles"]] + [_continuum(s, **block["continuum"])]
                amplitude = np.dot(np.column_stack(terms)[0], local)
                values[route.response_block_id] = flux * float(abs(amplitude) ** 2) / normalization
            return values
        edge = self.support_gev[0]

        if mass == edge:
            return self.edge_fraction_anchors
        coordinate = (mass - edge) / (RESPONSE_SEAM_JOIN_GEV - edge)
        one_minus = 1.0 - coordinate
        weights = (one_minus**3, 3.0 * one_minus**2 * coordinate, 3.0 * one_minus * coordinate**2, coordinate**3)
        values = {}

        for route in self.routes:
            block_id = route.response_block_id
            value = math.fsum(weight * control[block_id]
                              for weight, control in zip(weights, self.seam_simplex_controls, strict=True))
            if not math.isfinite(value) or value < -64.0 * math.ulp(1.0):
                raise EmResponseError(
                    f"cubic Bernstein response seam is negative for {block_id} at {mass:.12g} GeV")
            values[block_id] = value

        return values

    def rate_point_at(self, mass_gev: float, *, inclusive_width_gev: float,
                      external_width_gev: float) -> PostEdgeRatePoint:
        mass = float(mass_gev)

        if not self.support_gev[0] <= mass <= self.support_gev[1]:
            raise EmResponseError("rate-point mass lies outside production support")
        active = inclusive_width_gev - external_width_gev

        if active <= 0.0:
            raise EmResponseError("post-edge active width is nonpositive")
        resolved = {block_id: fraction * inclusive_width_gev
                    for block_id, fraction in self.response_fractions_at(mass).items()}
        subtotal = math.fsum(resolved.values())
        residual = active - subtotal

        if residual < -floating_tolerance(active, subtotal):
            raise EmResponseError(
                "fitted responses plus fixed external widths oversubscribe the inclusive uds width "
                f"at {mass:.12g} GeV by {-residual:.6g} GeV")

        return PostEdgeRatePoint(mass, active, resolved, max(residual, 0.0))
