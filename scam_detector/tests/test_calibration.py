"""
test_calibration.py
-------------------
Unit tests for IsotonicRegression score calibration module.
"""

import numpy as np
import pytest
from scam_detector.config import Config
from scam_detector.scoring.calibration import ScoreCalibrator


def test_calibrator_fallback_when_unfitted(tmp_path):
    calibrator = ScoreCalibrator(model_path=tmp_path / "nonexistent.joblib")
    assert not calibrator.is_fitted
    assert calibrator.calibrate(0.65) == 0.65
    assert calibrator.calibrate_batch([0.1, 0.5, 0.9]) == [0.1, 0.5, 0.9]


def test_calibrator_fit_and_calibrate(tmp_path):
    raw_scores = [0.1, 0.2, 0.35, 0.4, 0.6, 0.7, 0.8, 0.9] * 10
    labels = [0, 0, 0, 0, 1, 1, 1, 1] * 10

    calibrator = ScoreCalibrator(model_path=tmp_path / "calibrator.joblib", min_samples=10)
    calibrator.fit(raw_scores, labels)

    assert calibrator.is_fitted
    low_cal = calibrator.calibrate(0.15)
    high_cal = calibrator.calibrate(0.85)

    assert 0.0 <= low_cal <= high_cal <= 1.0
    assert low_cal < 0.3
    assert high_cal > 0.7


def test_calibrator_save_and_load(tmp_path):
    raw_scores = [0.1 * i for i in range(10)] * 5
    labels = [0 if s < 0.5 else 1 for s in raw_scores]

    model_file = tmp_path / "calibrator.joblib"
    calibrator = ScoreCalibrator(model_path=model_file, min_samples=10)
    calibrator.fit(raw_scores, labels)
    calibrator.save()

    assert model_file.exists()

    loaded_calibrator = ScoreCalibrator(model_path=model_file)
    assert loaded_calibrator.load()
    assert loaded_calibrator.is_fitted
    assert abs(loaded_calibrator.calibrate(0.2) - calibrator.calibrate(0.2)) < 1e-5


def test_calibrator_insufficient_samples():
    calibrator = ScoreCalibrator(min_samples=20)
    with pytest.raises(ValueError, match="Insufficient samples"):
        calibrator.fit([0.1, 0.9], [0, 1])
