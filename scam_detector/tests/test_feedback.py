"""
test_feedback.py
----------------
Tests for the human-review feedback store and label builder.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from scam_detector.feedback import (
    FeedbackStore,
    ReviewFeedback,
    build_training_labels,
    train_supervised_model,
)


def test_record_and_load_feedback(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "feedback.jsonl")
    fb = store.record_feedback(
        {
            "record_id": "rec-1",
            "reviewer_decision": "confirmed_scam",
            "reviewer_notes": "asks for Aadhaar + deposit",
        }
    )
    assert isinstance(fb, ReviewFeedback)
    assert fb.record_id == "rec-1"

    store.record_feedback(
        ReviewFeedback(
            record_id="rec-2",
            reviewer_decision="false_positive",
            reviewer_notes="legit campus drive",
        )
    )

    history = store.load_feedback_history()
    assert len(history) == 2
    assert history[0].reviewer_decision == "confirmed_scam"
    assert history[1].reviewer_decision == "false_positive"


def test_load_missing_file_returns_empty(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "missing.jsonl")
    assert store.load_feedback_history() == []


def test_build_training_labels_maps_and_dedupes() -> None:
    t0 = datetime(2026, 5, 1, tzinfo=timezone.utc)
    history = [
        ReviewFeedback(
            record_id="a",
            reviewer_decision="confirmed_legit",
            timestamp=t0,
        ),
        ReviewFeedback(
            record_id="a",
            reviewer_decision="confirmed_scam",
            timestamp=t0 + timedelta(hours=2),
            reviewer_notes="overturned after second look",
        ),
        ReviewFeedback(
            record_id="b",
            reviewer_decision="false_positive",
            timestamp=t0 + timedelta(hours=1),
        ),
    ]
    df = build_training_labels(history)
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) >= {
        "record_id",
        "label",
        "reviewer_decision",
        "reviewer_notes",
        "timestamp",
    }
    # Latest decision for "a" wins → scam label 1
    row_a = df.loc[df["record_id"] == "a"].iloc[0]
    assert int(row_a["label"]) == 1
    assert row_a["reviewer_decision"] == "confirmed_scam"
    row_b = df.loc[df["record_id"] == "b"].iloc[0]
    assert int(row_b["label"]) == 0
    assert len(df) == 2


def test_build_training_labels_empty() -> None:
    df = build_training_labels([])
    assert df.empty
    assert "label" in df.columns


def test_train_supervised_model_is_stub() -> None:
    df = build_training_labels(
        [
            ReviewFeedback(
                record_id="x",
                reviewer_decision="confirmed_scam",
            )
        ]
    )
    with pytest.raises(NotImplementedError, match="200"):
        train_supervised_model(df)
