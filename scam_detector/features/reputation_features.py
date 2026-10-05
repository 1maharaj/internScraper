"""
reputation_features.py
----------------------
Stateful tracking of company behavior across multiple pipeline runs over time.
This module implements the ReputationStore, reputation scoring, and feedback integration.

NOTE: This feature will be genuinely inert/no-op for the first several runs on
new data until enough history accumulates — that's expected, not a bug.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from scam_detector.feedback import FeedbackStore

log = logging.getLogger("scam_detector.features.reputation")


class CompanyReputation(BaseModel):
    """Aggregated reputation metrics for a single company."""

    company: str  # Lowercase normalized company name
    first_seen: str  # YYYY-MM-DD date string
    total_postings: int
    clear_count: int = 0
    review_count: int = 0
    block_count: int = 0
    average_scam_score: float = 0.0
    record_ids: list[str] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ReputationStore:
    """
    Append-only JSONL store for company reputation history.
    Saves the aggregated reputation stats per company.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def get_all_reputations(self) -> dict[str, CompanyReputation]:
        """Load all company reputations, returning a map of company_key -> CompanyReputation."""
        if not self.path.exists():
            return {}

        reputations: dict[str, CompanyReputation] = {}
        with self.path.open(encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    rep = CompanyReputation.model_validate_json(line)
                    reputations[rep.company] = rep
                except Exception as exc:
                    log.warning(
                        "Skipping malformed reputation at %s:%d (%s)",
                        self.path,
                        lineno,
                        exc,
                    )
        return reputations

    def update_reputations(self, updates: list[CompanyReputation]) -> None:
        """Append updated CompanyReputation records to the JSONL file."""
        with self.path.open("a", encoding="utf-8") as fh:
            for rep in updates:
                fh.write(rep.model_dump_json() + "\n")


def company_reputation_score(
    company: str,
    reputation_store: ReputationStore,
    feedback_store: FeedbackStore | None = None,
) -> float | None:
    """
    Calculate a reputation risk score for a company in [0, 1].

    Returns None for first-seen companies (no history).
    Otherwise returns a score:
    - 0.0 for consistently clear history (trust bonus).
    - 1.0 for a history of confirmed scams in the FeedbackStore.
    - Blended risk score for mixed decision histories.
    """
    company_key = (company or "").strip().lower()
    if not company_key:
        return None

    # Load history
    all_reps = reputation_store.get_all_reputations()
    if company_key not in all_reps:
        return None

    rep = all_reps[company_key]
    if rep.total_postings <= 0:
        return None

    # Check FeedbackStore for confirmed scams
    if feedback_store is not None:
        try:
            feedback_history = feedback_store.load_feedback_history()
            company_record_ids = set(rep.record_ids)
            for fb in feedback_history:
                if (
                    fb.reviewer_decision == "confirmed_scam"
                    and fb.record_id in company_record_ids
                ):
                    return 1.0
        except Exception as exc:
            log.warning(
                "Failed to check feedback store for company reputation: %s",
                exc,
            )

    # Otherwise, calculate reputation score from track record:
    # Consistently clear history (all past postings are clear) -> 0.0
    # Blend decision risk and average scam score.
    decision_risk = (
        rep.review_count * 0.5 + rep.block_count * 1.0
    ) / rep.total_postings
    avg_score_risk = rep.average_scam_score / 100.0
    reputation_score = 0.5 * decision_risk + 0.5 * avg_score_risk

    return round(max(0.0, min(1.0, reputation_score)), 4)
