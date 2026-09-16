"""Shared conventions of the absolute-width inputs of the matched hadronization model.

Providers supply inclusive and exclusive partial widths; the unowned
``other`` width is never supplied, it is computed by subtraction.
"""

from __future__ import annotations
import collections
from collections.abc import Sequence
import math

EXTERNAL_TREATMENT = "external"

RESOLVED_TREATMENT = "resolved"


class InputSchemaError(ValueError):
    """Raised when a physics input is malformed or mutually inconsistent."""


class InputSupportError(InputSchemaError):
    """Raised when a requested mass lies outside a closed input domain."""


def floating_tolerance(*values: float) -> float:
    """Return 16 ulp of the largest magnitude, a tolerance for floating arithmetic only."""

    scale = max((abs(float(value)) for value in values), default=0.0)
    return 16 * math.ulp(scale) if scale != 0.0 else 0.0


def final_state_key_from_pdgs(pdg_ids: Sequence[int]) -> str:
    """Return the canonical exclusive-support key for one particle list."""

    if not isinstance(pdg_ids, Sequence) or isinstance(pdg_ids, (str, bytes)):
        raise TypeError("PDG identifiers must be a sequence")
    if not pdg_ids or any(isinstance(pdg, bool) or not isinstance(pdg, int) or pdg == 0 for pdg in pdg_ids):
        raise InputSchemaError("PDG identifiers must be a nonempty list of nonzero integers")
    counts = collections.Counter(pdg_ids)
    return " ".join(f"{pdg_id}:{counts[pdg_id]}" for pdg_id in sorted(counts))
