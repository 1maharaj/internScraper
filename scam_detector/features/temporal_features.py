"""
temporal_features.py
--------------------
Signals derived from ``datePublished``, ``deadlineDate``, ``duration``,
and ``createdAt`` fields.

Design notes — Phase 1 alignment
----------------------------------
``deadline_urgency_score`` returns ``None`` (not ``0.0``) when
``deadlineDate`` is missing.  Phase 1 established that a null deadline means
*unknown*, not *no urgency*.  Callers must handle ``None`` explicitly —
treating it as zero urgency would produce systematic false negatives.

``posting_burst_score`` returns only the raw burst count and a cadence
dict.  It does NOT classify the burst as suspicious — that judgement requires
a reputation prior (Phase 6 / Prompt 6) because legitimate mass hiring drives
look identical to spam bursts in raw numbers alone.  The caller must combine
the burst count with the company reputation signal to make a decision.

``days_until_deadline`` / ``days_since_published`` are helpers for the
urgency feature and deadline_urgency_score respectively.

Date parsing
------------
Accepts ISO 8601 strings, datetime objects, date objects, and MongoDB
Extended JSON ``{"$date": "..."}`` dicts — via the shared ``_as_date``
coercion helper (same logic as data_quality.remediate).
"""

from __future__ import annotations

import math
from datetime import date, datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Pydantic output model
# ---------------------------------------------------------------------------


class TemporalFeatures(BaseModel):
    """Feature slice for date / timing signals."""

    deadline_expired: bool = Field(default=False)
    deadline_far_future: bool = Field(default=False)
    posting_future_dated: bool = Field(default=False)
    duration_implausible: bool = Field(default=False)
    days_to_deadline: float | None = Field(default=None)
    duration_months: float | None = Field(default=None)
    deadline_urgency_score: float | None = Field(
        default=None,
        description="None when deadline is absent (unknown, not zero urgency)",
    )
    posting_burst_count: int = Field(default=0, ge=0)
    posting_burst_cadence: float | None = Field(
        default=None,
        description="Mean days between postings in the window (historical cadence proxy)",
    )


# ---------------------------------------------------------------------------
# Date coercion helper
# ---------------------------------------------------------------------------


def _as_date(value: Any) -> date | None:
    """
    Coerce ISO strings, datetime/date objects, and MongoDB Extended JSON
    to a :class:`date`.  Returns ``None`` on any failure.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, dict):
        value = value.get("$date", "")
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(
                value.replace("Z", "+00:00")
            ).date()
        except (ValueError, TypeError):
            pass
    return None


def _today() -> date:
    """Return today's date in UTC.  Isolated so tests can monkeypatch."""
    return datetime.now(timezone.utc).date()


# ---------------------------------------------------------------------------
# Function 1 — posting_burst_score
# ---------------------------------------------------------------------------


def posting_burst_score(
    record: dict[str, Any],
    company_records: list[dict[str, Any]],
    window_days: int = 7,
) -> dict[str, Any]:
    """
    Count how many postings from the same company fall within a rolling
    *window_days* window centred on this record's ``datePublished``.

    Phase 1 caveat: a high burst count is NOT classified as suspicious here.
    Legitimate mass hiring drives (e.g. campus seasons) look identical to
    spam bursts.  Downstream risk scoring must combine this count with a
    company reputation prior (Phase 6 / Prompt 6) before flagging.

    Parameters
    ----------
    record:
        The internship being scored.  Must have ``datePublished`` and
        ``company`` keys.
    company_records:
        All postings by the **same** company (caller must pre-filter).
        The record itself may be included — it is counted naturally.
    window_days:
        Half-width of the rolling window in days.  A posting is "in-burst"
        if its datePublished is within ±window_days of the record's date.

    Returns
    -------
    dict with keys:
        ``burst_count``   — number of postings (incl. this one) within
                            the ±window_days window.
        ``cadence_days``  — mean inter-posting gap (days) across all
                            company_records sorted by date.  None when
                            fewer than 2 records have parseable dates.
                            Used as a historical cadence proxy by the
                            reputation prior in Phase 6.
    """
    target_date = _as_date(record.get("datePublished") or record.get("date_published"))

    if target_date is None or not company_records:
        return {"burst_count": 0, "cadence_days": None}

    half = window_days

    # Count in-window postings
    burst_count = 0
    for r in company_records:
        d = _as_date(r.get("datePublished") or r.get("date_published"))
        if d is not None and abs((d - target_date).days) <= half:
            burst_count += 1

    # Historical cadence: mean gap between consecutive dates across all records
    all_dates = sorted(
        d for r in company_records
        if (d := _as_date(r.get("datePublished") or r.get("date_published"))) is not None
    )
    if len(all_dates) >= 2:
        gaps = [(all_dates[i + 1] - all_dates[i]).days for i in range(len(all_dates) - 1)]
        cadence_days: float | None = round(sum(gaps) / len(gaps), 2)
    else:
        cadence_days = None

    return {"burst_count": burst_count, "cadence_days": cadence_days}


# ---------------------------------------------------------------------------
# Function 2 — deadline_urgency_score
# ---------------------------------------------------------------------------

# Thresholds for the urgency decay curve
_VERY_SHORT_DAYS = 3     # ≤3 days: maximum urgency pressure
_SHORT_DAYS = 14          # ≤14 days: high urgency
_MEDIUM_DAYS = 30         # ≤30 days: moderate urgency
_LONG_DAYS = 90           # >90 days: low urgency (near 0)


def deadline_urgency_score(record: dict[str, Any]) -> float | None:
    """
    Score the urgency pressure exerted by a short publication-to-deadline gap.

    Returns ``None`` (NOT ``0.0``) when ``deadlineDate`` is absent.  A
    missing deadline is semantically different from "no urgency" — it means
    unknown.  Callers must handle ``None`` explicitly.

    The score is a simple piecewise-linear decay from 1.0 (immediate deadline)
    to 0.0 (very distant deadline):

        ≤ 0 days (already expired)  → 1.0   (maximum pressure)
        1 – 3 days                  → 0.90 – 1.0
        4 – 14 days                 → 0.65 – 0.90
        15 – 30 days                → 0.35 – 0.65
        31 – 90 days                → 0.05 – 0.35
        > 90 days                   → 0.0 – 0.05

    Days are computed from ``datePublished`` to ``deadlineDate``, not from
    today — the goal is to capture the *designed* pressure of the posting,
    not how much time a candidate currently has.

    Returns
    -------
    float in [0, 1] or None
    """
    deadline = _as_date(record.get("deadlineDate"))
    if deadline is None:
        return None

    published = _as_date(record.get("datePublished") or record.get("date_published"))
    if published is None:
        # Fall back to today as reference so score is still computed
        published = _today()

    days_gap = (deadline - published).days

    if days_gap <= 0:
        return 1.0
    if days_gap <= _VERY_SHORT_DAYS:
        # Linear from 1.0 down to 0.90 over 0→3 days
        return round(1.0 - (days_gap / _VERY_SHORT_DAYS) * 0.10, 4)
    if days_gap <= _SHORT_DAYS:
        # Linear from 0.90 down to 0.65 over 3→14 days
        frac = (days_gap - _VERY_SHORT_DAYS) / (_SHORT_DAYS - _VERY_SHORT_DAYS)
        return round(0.90 - frac * 0.25, 4)
    if days_gap <= _MEDIUM_DAYS:
        # Linear from 0.65 down to 0.35 over 14→30 days
        frac = (days_gap - _SHORT_DAYS) / (_MEDIUM_DAYS - _SHORT_DAYS)
        return round(0.65 - frac * 0.30, 4)
    if days_gap <= _LONG_DAYS:
        # Linear from 0.35 down to 0.05 over 30→90 days
        frac = (days_gap - _MEDIUM_DAYS) / (_LONG_DAYS - _MEDIUM_DAYS)
        return round(0.35 - frac * 0.30, 4)

    # Exponential decay beyond 90 days
    excess = days_gap - _LONG_DAYS
    return round(max(0.05 * math.exp(-excess / 120), 0.0), 4)
