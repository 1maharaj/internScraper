"""
stipend_features.py
-------------------
Signals derived from the ``stipend`` and ``duration`` fields.

All public functions are pure and individually testable.

Design notes
------------
- Stipends are normalised to **INR per hour** as a common unit so records
  with monthly, weekly, and lump-sum amounts can be compared directly.
- ``performance_based`` stipends return ``None`` from normalisation — they
  are not comparable to fixed amounts and must be handled separately by the
  caller.  Do NOT substitute 0 for None here.
- ``unpaid`` stipends normalise to exactly ``0.0`` so they participate in
  z-score comparisons as the lower bound.
- Z-score uses a peer group supplied by the caller (pipeline controls grouping
  by field/tags/isRemote overlap).  The function never fetches its own peers.
- ``stipend_perk_consistency_check`` is a hard structural check (bool), not a
  weighted score.  A True return means the record contains internally
  contradictory data.

Currency conversion rates (INR)
---------------------------------
Approximate mid-market rates used for normalisation.  These are intentionally
hardcoded as constants to keep the function deterministic — a live FX lookup
would introduce non-determinism unsuitable for batch scoring.  Update the
``_FX_TO_INR`` table periodically or wire a real FX provider in production.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Pydantic output model (kept for backwards compat with FeatureVector)
# ---------------------------------------------------------------------------


class StipendFeatures(BaseModel):
    """Feature slice for stipend / compensation signals."""

    is_outlier_high: bool = Field(default=False)
    is_outlier_low: bool = Field(default=False)
    pay_to_work_score: float = Field(default=0.0, ge=0.0, le=1.0)
    missing_stipend_for_role: bool = Field(default=False)
    amount_plausibility_score: float = Field(default=1.0, ge=0.0, le=1.0)
    stipend_type: str = Field(default="unknown")
    # Extended fields populated by this module
    hourly_inr: float | None = Field(default=None)
    peer_zscore: float | None = Field(default=None)
    perk_consistency_ok: bool = Field(default=True)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Hours per period assumptions used for normalisation
_HOURS_PER_MONTH: float = 160.0   # 20 working days × 8 hours
_HOURS_PER_WEEK: float = 40.0
# lump-sum: amortise over the whole internship duration (computed dynamically)

# Approximate INR conversion rates (mid-market, as of mid-2025)
# Update periodically or wire to a real FX provider in production.
_FX_TO_INR: dict[str, float] = {
    "INR": 1.0,
    "USD": 83.5,
    "EUR": 90.0,
    "GBP": 105.0,
    "AED": 22.7,
    "SGD": 62.0,
    "CAD": 62.0,
    "AUD": 54.0,
    "JPY": 0.56,
}

# Perks that imply monetary compensation — used by consistency check
_PAID_PERKS: frozenset[str] = frozenset(
    {
        "stipend",
        "paid stipend",
        "monthly stipend",
        "performance-based stipend",
        "competitive stipend",
    }
)

# Z-score thresholds for outlier classification
_ZSCORE_HIGH = 2.5   # more than 2.5 SD above mean → suspiciously high
_ZSCORE_LOW = -2.0   # more than 2 SD below mean   → suspiciously low


# ---------------------------------------------------------------------------
# Function 1 — normalize_stipend_to_hourly_inr
# ---------------------------------------------------------------------------


def normalize_stipend_to_hourly_inr(
    stipend: dict[str, Any],
    duration: dict[str, Any],
) -> float | None:
    """
    Convert a stipend dict to a single comparable unit: **INR per hour**.

    Parameters
    ----------
    stipend:
        Dict with keys ``type``, ``amount``, ``currency``, ``period``.
        ``type`` must be one of: ``"paid"``, ``"unpaid"``,
        ``"performance-based"``, ``"performance_based"``.
    duration:
        Dict with keys ``value`` (number) and ``unit``
        (``"months"`` or ``"weeks"``).

    Returns
    -------
    float
        INR-per-hour equivalent.  Returns ``0.0`` for unpaid.
    None
        Returned for ``performance-based`` stipends (not a fixed comparable
        number) and when critical fields are missing/unparseable.

    Edge cases
    ----------
    - ``duration.value == 0``  → treated as 1 month to avoid divide-by-zero
    - Unknown currency         → treated as INR (conservative assumption)
    - Missing amount for paid  → returns None (amount unknown)
    """
    if not stipend or not isinstance(stipend, dict):
        return None

    stype = (stipend.get("type") or "").strip().lower().replace("-", "_")

    if stype == "unpaid":
        return 0.0

    if stype in ("performance_based", "performance based"):
        return None  # not a fixed comparable number — caller must handle

    if stype != "paid":
        return None  # unknown type — cannot normalise

    amount = stipend.get("amount")
    if amount is None or not isinstance(amount, (int, float)):
        return None
    amount = float(amount)

    currency = (stipend.get("currency") or "INR").strip().upper()
    fx_rate = _FX_TO_INR.get(currency, 1.0)   # unknown → treat as INR
    amount_inr = amount * fx_rate

    period = (stipend.get("period") or "monthly").strip().lower()

    if period == "monthly":
        return amount_inr / _HOURS_PER_MONTH

    if period == "weekly":
        return amount_inr / _HOURS_PER_WEEK

    if period == "lump-sum":
        # Amortise over total internship duration
        dur_value = float(duration.get("value") or 0)
        dur_unit = (duration.get("unit") or "months").strip().lower()

        if dur_value <= 0:
            dur_value = 1.0  # guard against zero/missing duration

        if dur_unit == "months":
            total_hours = dur_value * _HOURS_PER_MONTH
        elif dur_unit == "weeks":
            total_hours = dur_value * _HOURS_PER_WEEK
        else:
            total_hours = _HOURS_PER_MONTH  # unknown unit → assume 1 month

        return amount_inr / total_hours

    # Fallback: treat unknown period as monthly
    return amount_inr / _HOURS_PER_MONTH


# ---------------------------------------------------------------------------
# Function 2 — stipend_zscore
# ---------------------------------------------------------------------------


def stipend_zscore(
    record: dict[str, Any],
    peer_group_records: list[dict[str, Any]],
) -> float | None:
    """
    Compute a z-score for this record's normalised stipend against *peer_group_records*.

    Both extremely high and extremely low z-scores are suspicious:
    - High (> ``_ZSCORE_HIGH``)  → unrealistically attractive stipend
    - Low  (< ``_ZSCORE_LOW``)   → exploitatively low for the category

    The caller (``pipeline.py``) is responsible for building the peer group
    by filtering on field/tags overlap and isRemote.  This function never
    fetches or constructs peer groups itself.

    Parameters
    ----------
    record:
        Single internship dict with ``stipend`` and ``duration`` keys.
    peer_group_records:
        List of comparable internship dicts (same field / isRemote category).

    Returns
    -------
    float
        Z-score (can be negative for below-average stipends).
    None
        Returned when:
        - Fewer than 2 peers have normalisable stipends (no meaningful σ)
        - This record's own stipend is ``performance-based`` (not a number)
        - This record's own stipend is missing/unparseable
    """
    own_hourly = normalize_stipend_to_hourly_inr(
        record.get("stipend") or {},
        record.get("duration") or {},
    )
    if own_hourly is None:
        return None

    peer_hourlies: list[float] = []
    for r in peer_group_records:
        h = normalize_stipend_to_hourly_inr(
            r.get("stipend") or {},
            r.get("duration") or {},
        )
        if h is not None:
            peer_hourlies.append(h)

    if len(peer_hourlies) < 2:
        return None  # not enough data for a meaningful z-score

    mu = statistics.mean(peer_hourlies)
    sigma = statistics.stdev(peer_hourlies)

    if sigma == 0.0:
        # All peers have identical stipends; this record matches exactly → z=0
        return 0.0 if own_hourly == mu else (float("inf") if own_hourly > mu else float("-inf"))

    return round((own_hourly - mu) / sigma, 4)


# ---------------------------------------------------------------------------
# Function 3 — stipend_perk_consistency_check
# ---------------------------------------------------------------------------


def stipend_perk_consistency_check(record: dict[str, Any]) -> bool:
    """
    Hard structural consistency check between ``stipend.type`` and ``perks``.

    Returns ``True`` (inconsistency detected) when:
    - ``stipend.type == "unpaid"`` but a perk name implies monetary payment
      (e.g. perk = "Stipend", "Monthly Stipend", "Paid Stipend")

    Returns ``False`` (no inconsistency) otherwise, including when either
    field is missing.

    This is a data-quality signal, not a fraud signal on its own — it most
    commonly indicates a scraper that copied the perks list from the paid
    version of the same template while the stipend field was set to unpaid.

    Parameters
    ----------
    record:
        Internship dict with ``stipend`` and ``perks`` keys.

    Returns
    -------
    bool
        ``True`` → internal contradiction detected.
    """
    stipend = record.get("stipend")
    if isinstance(stipend, str):
        stype = stipend.strip().lower()
    elif isinstance(stipend, dict):
        stype = (stipend.get("type") or "").strip().lower()
    else:
        stype = ""


    if stype != "unpaid":
        return False  # only relevant when claimed unpaid

    perks_raw = record.get("perks") or []
    if not isinstance(perks_raw, list):
        return False

    for perk in perks_raw:
        if isinstance(perk, str) and perk.strip().lower() in _PAID_PERKS:
            return True

    return False
