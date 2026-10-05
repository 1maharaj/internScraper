"""
test_tune_threshold.py
----------------------
Unit tests for threshold tuning utility (tools/tune_threshold.py).
"""

import json
import pytest
import pandas as pd
from pathlib import Path

from scam_detector.tools.tune_threshold import (
    compute_pr_curve,
    find_optimal_threshold,
    load_labeled_dataset,
    apply_thresholds_to_config,
    ThresholdEvaluation,
)


def test_compute_pr_curve() -> None:
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    y_scores = [10.0, 20.0, 30.0, 40.0, 60.0, 70.0, 80.0, 90.0]

    curve = compute_pr_curve(y_true, y_scores, step=10.0)
    assert len(curve) > 0

    # At threshold 50.0: TP=4, FP=0, TN=4, FN=0 -> Precision=1.0, Recall=1.0, F1=1.0
    eval_50 = next(e for e in curve if abs(e.threshold - 50.0) < 1e-3)
    assert eval_50.precision == 1.0
    assert eval_50.recall == 1.0
    assert eval_50.f1_score == 1.0
    assert eval_50.tp == 4
    assert eval_50.fp == 0


def test_find_optimal_threshold_max_f1() -> None:
    curve = [
        ThresholdEvaluation(threshold=20.0, precision=0.5, recall=1.0, f1_score=0.67, tp=4, fp=4, tn=0, fn=0),
        ThresholdEvaluation(threshold=50.0, precision=1.0, recall=1.0, f1_score=1.00, tp=4, fp=0, tn=4, fn=0),
        ThresholdEvaluation(threshold=80.0, precision=1.0, recall=0.5, f1_score=0.67, tp=2, fp=0, tn=4, fn=2),
    ]

    opt = find_optimal_threshold(curve)
    assert opt.threshold == 50.0
    assert opt.f1_score == 1.00


def test_find_optimal_threshold_target_precision() -> None:
    curve = [
        ThresholdEvaluation(threshold=20.0, precision=0.6, recall=1.0, f1_score=0.75, tp=4, fp=2, tn=2, fn=0),
        ThresholdEvaluation(threshold=50.0, precision=0.85, recall=0.75, f1_score=0.80, tp=3, fp=1, tn=3, fn=1),
        ThresholdEvaluation(threshold=80.0, precision=0.95, recall=0.5, f1_score=0.65, tp=2, fp=0, tn=4, fn=2),
    ]

    opt = find_optimal_threshold(curve, target_precision=0.80)
    assert opt.precision >= 0.80


def test_load_labeled_dataset_json(tmp_path: Path) -> None:
    data = [
        {"record_id": "r1", "label": 1, "scam_score": 85.0},
        {"record_id": "r2", "label": 0, "scam_score": 15.0},
        {"record_id": "r3", "reviewer_decision": "confirmed_scam", "scam_score": 90.0},
        {"record_id": "r4", "reviewer_decision": "confirmed_legit", "scam_score": 5.0},
    ]
    file_path = tmp_path / "dataset.json"
    with open(file_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)

    y_true, y_scores = load_labeled_dataset(file_path)
    assert len(y_true) == 4
    assert len(y_scores) == 4
    assert y_true == [1, 0, 1, 0]
    assert y_scores == [85.0, 15.0, 90.0, 5.0]


def test_apply_thresholds_to_config(tmp_path: Path) -> None:
    sample_config = '''
class DecisionThresholds(BaseModel):
    clear_below: float = Field(
        default=30.0,
        ge=0.0,
        le=100.0,
    )
    block_at_or_above: float = Field(
        default=70.0,
        ge=0.0,
        le=100.0,
    )
'''
    config_file = tmp_path / "config.py"
    config_file.write_text(sample_config, encoding="utf-8")

    apply_thresholds_to_config(block_threshold=82.5, review_threshold=25.0, config_path=config_file)

    updated = config_file.read_text(encoding="utf-8")
    assert "default=82.5" in updated
    assert "default=25.0" in updated
