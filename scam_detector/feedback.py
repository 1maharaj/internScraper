"""
feedback.py
-----------
Human-review feedback loop for Phase 1 / Phase 6 "review"-bucket decisions.

Reviewers confirm or overturn pipeline decisions.  Accumulated labels are
later convertible into a supervised training set that can replace or
augment the unsupervised IsolationForest from Prompt 7.

This module is intentionally **not** wired into ``pipeline.run_pipeline``.
Only start collecting feedback once Prompts 0–9 are live and real review
decisions exist; only train a supervised model once ~200+ balanced labels
are available (see ``train_supervised_model``).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

log = logging.getLogger("scam_detector.feedback")

ReviewerDecision = Literal["confirmed_scam", "false_positive", "confirmed_legit"]

# Binary label mapping for supervised training:
#   confirmed_scam  → 1 (positive / scam class)
#   false_positive  → 0 (reviewer overturned a scam-ish signal → legit)
#   confirmed_legit → 0 (negative / legit class)
_LABEL_MAP: dict[str, int] = {
    "confirmed_scam": 1,
    "false_positive": 0,
    "confirmed_legit": 0,
}

_MIN_LABELED_EXAMPLES = 200


class ReviewFeedback(BaseModel):
    """One human-review outcome for a single internship record."""

    record_id: str = Field(description="Stable id of the reviewed internship")
    reviewer_decision: ReviewerDecision = Field(
        description=(
            "confirmed_scam — pipeline was right / listing is fraudulent; "
            "false_positive — pipeline over-flagged a legit listing; "
            "confirmed_legit — listing is legitimate (e.g. cleared from review)"
        )
    )
    reviewer_notes: str = Field(default="", description="Free-text reviewer rationale")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC time the feedback was recorded",
    )


class FeedbackStore:
    """
    Append-only JSONL store for review feedback.

    No database yet — a local file is enough while review volume is low.
    Each line is one serialised :class:`ReviewFeedback` object.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record_feedback(self, feedback: ReviewFeedback | dict[str, Any]) -> ReviewFeedback:
        """
        Append one feedback event to the JSONL file.

        Accepts a :class:`ReviewFeedback` instance or a raw dict (validated).
        Returns the normalised model that was written.
        """
        event = (
            feedback
            if isinstance(feedback, ReviewFeedback)
            else ReviewFeedback.model_validate(feedback)
        )
        line = event.model_dump_json() + "\n"
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line)
        log.info(
            "Recorded feedback for %s → %s",
            event.record_id,
            event.reviewer_decision,
        )
        return event

    def load_feedback_history(self) -> list[ReviewFeedback]:
        """
        Load all feedback events from the JSONL file (oldest first).

        Malformed lines are skipped with a warning so a single bad row
        cannot wipe the whole history.
        """
        if not self.path.exists():
            return []

        history: list[ReviewFeedback] = []
        with self.path.open(encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    history.append(ReviewFeedback.model_validate_json(line))
                except Exception as exc:
                    log.warning(
                        "Skipping malformed feedback at %s:%d (%s)",
                        self.path,
                        lineno,
                        exc,
                    )
        return history


def build_training_labels(feedback_history: list[ReviewFeedback]) -> pd.DataFrame:
    """
    Turn accumulated feedback into a labeled DataFrame for supervised training.

    Columns
    -------
    record_id : str
    label : int
        ``1`` = scam (``confirmed_scam``), ``0`` = legit
        (``false_positive`` or ``confirmed_legit``).
    reviewer_decision : str
    reviewer_notes : str
    timestamp : datetime

    If the same ``record_id`` appears more than once, the **latest** feedback
    by timestamp wins (reviewers may overturn earlier decisions).
    """
    if not feedback_history:
        return pd.DataFrame(
            columns=[
                "record_id",
                "label",
                "reviewer_decision",
                "reviewer_notes",
                "timestamp",
            ]
        )

    rows = [
        {
            "record_id": fb.record_id,
            "label": _LABEL_MAP[fb.reviewer_decision],
            "reviewer_decision": fb.reviewer_decision,
            "reviewer_notes": fb.reviewer_notes,
            "timestamp": fb.timestamp,
        }
        for fb in feedback_history
    ]
    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").drop_duplicates(subset=["record_id"], keep="last")
    df = df.reset_index(drop=True)
    return df


def train_supervised_model(labeled_df: pd.DataFrame) -> None:
    """
    Stub: train a supervised scam classifier from human-labeled feedback.

    Intended eventual behaviour
    ---------------------------
    Fit logistic regression or gradient boosting on feature matrices joined
    to ``labeled_df`` labels, then serialise the artifact under
    ``scam_detector/models/`` for use by the risk engine — replacing or
    augmenting the unsupervised IsolationForest from Prompt 7.

    Gate — do **not** call this in production until all of the following hold:
      • at least ~200 labeled examples (``len(labeled_df) >= 200``)
      • reasonable class balance (neither scam nor legit below ~20% of rows)
      • feature matrix join is available for the labeled ``record_id``s

    This function is **not** wired into ``pipeline.run_pipeline``.  Keep it
    offline / notebook / admin-CLI only until the gates above are met.
    """
    n = 0 if labeled_df is None else len(labeled_df)
    raise NotImplementedError(
        "train_supervised_model is a stub.  Do not run until you have at least "
        f"~{_MIN_LABELED_EXAMPLES} labeled examples with reasonable class balance "
        f"(currently n={n}).  See module docstring in scam_detector.feedback."
    )
