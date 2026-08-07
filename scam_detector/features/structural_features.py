"""
structural_features.py
----------------------
Signals derived from the structural completeness of a listing.

Design notes
------------
``field_completeness_score`` measures information density, NOT fraud risk.
Sparse records aren't necessarily fraudulent — they may simply be low-effort
or early-stage postings.  This score should feed into a *confidence* signal
in the risk engine (Prompt 8) that down-weights other features when the
record lacks enough information to score reliably.  Do NOT use it directly
as a fraud signal.

``openings_zscore`` follows the same peer-group pattern as
``stipend_zscore`` — the caller supplies the peer group.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Pydantic output model
# ---------------------------------------------------------------------------


class StructuralFeatures(BaseModel):
    """Feature slice for listing-structure / completeness signals."""

    completeness_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    skills_count: int = Field(default=0, ge=0)
    skills_count_anomaly: bool = Field(default=False)
    desc_to_title_ratio: float = Field(default=0.0, ge=0.0)
    has_contact_in_body: bool = Field(default=False)
    responsibilities_count: int = Field(default=0, ge=0)
    # Extended fields
    openings_zscore: float | None = Field(default=None)
    field_completeness: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Information-density score (0–1).  "
            "Feeds into confidence, NOT fraud score directly."
        ),
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Skills count thresholds for anomaly detection
_SKILLS_MIN = 1     # fewer than this → likely no skills listed
_SKILLS_MAX = 30    # more than this → suspiciously long / copy-pasted list

# Openings z-score thresholds (same direction logic as stipend)
_OPENINGS_ZSCORE_HIGH = 3.0   # very large batch posting
_OPENINGS_ZSCORE_LOW = -2.0   # not meaningful for openings (negative always ok)

# Fields checked by field_completeness_score
# Each is (field_key, check_fn) where check_fn returns True if the field
# is meaningfully populated.
def _non_empty_list(v: Any) -> bool:
    return isinstance(v, list) and len(v) > 0

def _non_empty_str(v: Any) -> bool:
    return isinstance(v, str) and bool(v.strip())

_COMPLETENESS_CHECKS: list[tuple[str, Any]] = [
    ("skills",           _non_empty_list),
    ("field",            _non_empty_list),
    ("responsibilities", _non_empty_list),
    ("perks",            _non_empty_list),
    ("degree",           _non_empty_list),
    ("tags",             _non_empty_list),
    ("summary",          _non_empty_str),
    ("city",             _non_empty_str),
]


# ---------------------------------------------------------------------------
# Function 1 — openings_zscore
# ---------------------------------------------------------------------------


def openings_zscore(
    record: dict[str, Any],
    peer_group_records: list[dict[str, Any]],
) -> float | None:
    """
    Compute a z-score for this record's ``openings`` count against peers.

    An abnormally high ``openings`` count can indicate a shell company posting
    fictitious mass vacancies.  Very low counts (1–2) are normal and not
    suspicious.

    The caller is responsible for building the peer group by field/tags/
    isRemote overlap.  This function never fetches its own peers.

    Parameters
    ----------
    record:
        Internship dict with ``openings`` key (int or None).
    peer_group_records:
        List of comparable internship dicts.

    Returns
    -------
    float or None
        None when:
        - This record's openings is None/missing
        - Fewer than 2 peers have non-None openings values
        - All peers have identical openings (σ = 0, would divide by zero)
    """
    own = record.get("openings")
    if own is None or not isinstance(own, (int, float)):
        return None
    own_f = float(own)

    peer_vals: list[float] = []
    for r in peer_group_records:
        v = r.get("openings")
        if v is not None and isinstance(v, (int, float)):
            peer_vals.append(float(v))

    if len(peer_vals) < 2:
        return None

    mu = statistics.mean(peer_vals)
    sigma = statistics.stdev(peer_vals)

    if sigma == 0.0:
        return 0.0 if own_f == mu else None

    return round((own_f - mu) / sigma, 4)


# ---------------------------------------------------------------------------
# Function 2 — field_completeness_score
# ---------------------------------------------------------------------------


def field_completeness_score(record: dict[str, Any]) -> float:
    """
    Fraction of optional-but-informative fields that are meaningfully
    populated.

    Checks: ``skills``, ``field``, ``responsibilities``, ``perks``,
    ``degree``, ``tags``, ``summary``, ``city``.

    Returns
    -------
    float in [0, 1]
        1.0 → all checked fields populated.
        0.0 → none populated.

    ⚠  Design note: this score measures *confidence*, not *fraud risk*.
    Use it to down-weight other features when the record lacks enough
    information to score reliably.  Do NOT treat a low score as a direct
    fraud signal.
    """
    if not record:
        return 0.0

    passed = sum(
        1 for key, check in _COMPLETENESS_CHECKS
        if check(record.get(key))
    )
    return round(passed / len(_COMPLETENESS_CHECKS), 4)
