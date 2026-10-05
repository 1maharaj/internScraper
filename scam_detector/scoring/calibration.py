"""
calibration.py
--------------
Isotonic regression score calibration module for scam_detector.

Fits an IsotonicRegression model on raw blended risk scores [0.0, 1.0] against
labeled feedback outcomes (1 = scam, 0 = legit) to produce a well-calibrated
posterior scam probability score.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from scam_detector.config import Config, cfg as default_cfg
from scam_detector.feedback import FeedbackStore, build_training_labels

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

log = logging.getLogger("scam_detector.scoring.calibration")

MODEL_VERSION = "1.0.0"
DEFAULT_MIN_CALIBRATION_SAMPLES = 50


class ScoreCalibrator:
    """
    Isotonic regression score calibrator wrapper.

    Maps uncalibrated raw blended scores [0.0, 1.0] into calibrated scam probabilities.
    Falls back gracefully to raw scores when no trained model is present.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        min_samples: int = DEFAULT_MIN_CALIBRATION_SAMPLES,
        config: Config | None = None,
    ) -> None:
        self.config = config or default_cfg
        self.min_samples = min_samples
        self.model_path = Path(model_path) if model_path else Path(
            getattr(getattr(self.config, "calibration", None), "model_path", "scam_detector/models/calibration_model.joblib")
        )
        self._model: IsotonicRegression | None = None
        self._model_version: str = MODEL_VERSION
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def model_version(self) -> str:
        return self._model_version

    def fit(
        self,
        raw_scores: Sequence[float],
        labels: Sequence[int],
        *,
        min_samples: int | None = None,
    ) -> ScoreCalibrator:
        """
        Fit IsotonicRegression model on raw scores [0.0, 1.0] and binary labels [0, 1].
        """
        threshold = min_samples if min_samples is not None else self.min_samples
        if len(raw_scores) != len(labels):
            raise ValueError(
                f"Mismatch between raw_scores length ({len(raw_scores)}) and labels length ({len(labels)})"
            )

        if len(raw_scores) < threshold:
            raise ValueError(
                f"Insufficient samples to fit calibrator: found {len(raw_scores)}, required at least {threshold}"
            )

        X = np.clip(np.asarray(raw_scores, dtype=float), 0.0, 1.0)
        y = np.asarray(labels, dtype=int)

        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(X, y)

        self._model = iso
        self._model_version = MODEL_VERSION
        self._is_fitted = True

        log.info("Fitted IsotonicRegression calibrator on %d samples", len(X))
        return self

    def calibrate(self, raw_score: float) -> float:
        """
        Calibrate a single raw score in [0.0, 1.0].
        Returns raw_score unchanged if calibrator is not fitted and cannot be loaded.
        """
        if not self._is_fitted and not self.load():
            return float(max(0.0, min(1.0, raw_score)))

        if self._model is None:
            return float(max(0.0, min(1.0, raw_score)))

        score_arr = np.array([max(0.0, min(1.0, raw_score))], dtype=float)
        calibrated = self._model.transform(score_arr)[0]
        return float(max(0.0, min(1.0, calibrated)))

    def calibrate_batch(self, raw_scores: Sequence[float]) -> list[float]:
        """Calibrate a sequence of raw scores in [0.0, 1.0]."""
        if not raw_scores:
            return []

        if not self._is_fitted and not self.load():
            return [float(max(0.0, min(1.0, s))) for s in raw_scores]

        if self._model is None:
            return [float(max(0.0, min(1.0, s))) for s in raw_scores]

        X = np.clip(np.asarray(raw_scores, dtype=float), 0.0, 1.0)
        calibrated = self._model.transform(X)
        return [float(max(0.0, min(1.0, c))) for c in calibrated]

    def save(self, path: str | Path | None = None) -> None:
        """Serialize calibrator model artifact to disk via joblib."""
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Cannot save an unfitted ScoreCalibrator")

        target_path = Path(path) if path else self.model_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        bundle = {
            "model": self._model,
            "version": self._model_version,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        if not HAS_JOBLIB:
            raise ImportError("joblib is required to save ScoreCalibrator")

        joblib.dump(bundle, target_path)
        log.info("Saved ScoreCalibrator artifact to %s", target_path)

    def load(self, path: str | Path | None = None) -> bool:
        """
        Deserialize calibrator model artifact from disk via joblib.
        Returns True if successfully loaded, False otherwise.
        """
        target_path = Path(path) if path else self.model_path
        if not target_path.exists() or not HAS_JOBLIB:
            self._is_fitted = False
            return False

        try:
            bundle = joblib.load(target_path)
            if not isinstance(bundle, dict) or "model" not in bundle:
                log.warning("Invalid calibrator bundle in %s", target_path)
                self._is_fitted = False
                return False

            self._model = bundle["model"]
            self._model_version = bundle.get("version", MODEL_VERSION)
            self._is_fitted = True
            log.info("Loaded ScoreCalibrator (v%s) from %s", self._model_version, target_path)
            return True
        except Exception as exc:
            log.warning("Failed to load ScoreCalibrator from %s: %s", target_path, exc)
            self._is_fitted = False
            return False


def refit_calibration_from_feedback(
    feedback_path: str | Path,
    model_path: str | Path | None = None,
    raw_scores_map: dict[str, float] | None = None,
    min_samples: int = DEFAULT_MIN_CALIBRATION_SAMPLES,
) -> ScoreCalibrator:
    """
    CLI / background refitting helper. Reads feedback history, pairs record labels with
    their raw scores, fits ScoreCalibrator, and saves it.
    """
    store = FeedbackStore(feedback_path)
    history = store.load_feedback_history()
    labeled_df = build_training_labels(history)

    if labeled_df.empty:
        raise ValueError("No labeled feedback history found in " + str(feedback_path))

    if raw_scores_map is not None:
        labeled_df["raw_score"] = labeled_df["record_id"].map(raw_scores_map)
        labeled_df = labeled_df.dropna(subset=["raw_score"])

    if "raw_score" not in labeled_df.columns:
        raise ValueError(
            "labeled_df missing raw_score mapping. Provide raw_scores_map for feedback records."
        )

    raw_scores = labeled_df["raw_score"].tolist()
    labels = labeled_df["label"].tolist()

    calibrator = ScoreCalibrator(model_path=model_path, min_samples=min_samples)
    calibrator.fit(raw_scores, labels)
    calibrator.save()
    return calibrator


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Refit Isotonic Score Calibrator from FeedbackStore")
    parser.add_argument("--feedback-path", type=str, required=True, help="Path to feedback JSONL store")
    parser.add_argument("--scores-json", type=str, required=True, help="Path to JSON file mapping record_id to raw_score")
    parser.add_argument("--model-path", type=str, default=None, help="Output path for joblib model artifact")
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_CALIBRATION_SAMPLES, help="Min labeled samples required")

    args = parser.parse_args()

    with open(args.scores_json, encoding="utf-8") as fh:
        raw_scores_map = json.load(fh)

    refit_calibration_from_feedback(
        feedback_path=args.feedback_path,
        model_path=args.model_path,
        raw_scores_map=raw_scores_map,
        min_samples=args.min_samples,
    )
    print("Score calibration refit complete.")


if __name__ == "__main__":
    main()
