"""
supervised_model.py
-------------------
Supervised machine learning scoring layer for scam_detector.

Fits a Gradient-Boosted Classifier (LightGBM if available, fallback to
sklearn's GradientBoostingClassifier) on labeled feedback data produced by
`feedback.build_training_labels()`.

Reuses feature matrices assembled by `scoring.anomaly_model.assemble_feature_matrix`
to avoid duplicating feature extraction logic.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from scam_detector.config import Config, cfg as default_cfg
from scam_detector.feedback import ReviewFeedback, build_training_labels
from scam_detector.scoring.anomaly_model import assemble_feature_matrix

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from sklearn.ensemble import GradientBoostingClassifier

log = logging.getLogger("scam_detector.scoring.supervised_model")

MODEL_VERSION = "1.0.0"
DEFAULT_MIN_TRAINING_SAMPLES = 500


class SupervisedResult(BaseModel):
    """Output container for supervised model inference."""

    model_available: bool = Field(
        default=False,
        description="True if a trained supervised model was loaded and evaluated",
    )
    scam_probability: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Predicted scam probability [0.0, 1.0], or None if model unavailable",
    )
    model_version: str = Field(default="none", description="Version stamp of the model artifact")
    backend: str = Field(default="none", description="Classifier backend ('lightgbm' or 'sklearn')")


class SupervisedScamModel:
    """
    Supervised scam classification model wrapper.

    Handles training, joblib serialization/deserialization with version stamps,
    and inference over feature DataFrames.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        min_training_samples: int = DEFAULT_MIN_TRAINING_SAMPLES,
        config: Config | None = None,
    ) -> None:
        self.config = config or default_cfg
        self.min_training_samples = min_training_samples
        self.model_path = Path(model_path) if model_path else Path(
            getattr(getattr(self.config, "supervised", None), "model_path", "scam_detector/models/supervised_model.joblib")
        )
        self._model: Any = None
        self._feature_columns: list[str] = []
        self._model_version: str = MODEL_VERSION
        self._backend: str = "none"
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def feature_columns(self) -> list[str]:
        return list(self._feature_columns)

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def backend(self) -> str:
        return self._backend

    def fit(
        self,
        records: Sequence[Any],
        feature_vectors: Sequence[Any],
        feedback_history: list[ReviewFeedback],
        *,
        min_samples: int | None = None,
    ) -> SupervisedScamModel:
        """
        Train the supervised model on labeled feedback history joined with
        assembled feature matrices.
        """
        threshold = min_samples if min_samples is not None else self.min_training_samples
        labeled_df = build_training_labels(feedback_history)

        if labeled_df.empty or len(labeled_df) < threshold:
            n_samples = len(labeled_df)
            log.warning(
                "Supervised training skipped: labeled samples (%d) below MIN_TRAINING_SAMPLES (%d)",
                n_samples,
                threshold,
            )
            raise ValueError(
                f"Insufficient labeled samples to train supervised model: "
                f"found {n_samples}, required at least {threshold}"
            )

        # Assemble feature matrix for provided records & vectors
        df_features = assemble_feature_matrix(records, feature_vectors)
        if df_features.empty:
            raise ValueError("Cannot train supervised model on an empty feature matrix")

        # Join labeled_df with df_features by record_id
        merged = labeled_df.merge(
            df_features, left_on="record_id", right_index=True, how="inner"
        )

        if len(merged) < threshold:
            log.warning(
                "Supervised training skipped: matching feature rows (%d) below required (%d)",
                len(merged),
                threshold,
            )
            raise ValueError(
                f"Insufficient matching feature rows for labeled records: "
                f"found {len(merged)}, required at least {threshold}"
            )

        feature_cols = [c for c in df_features.columns]
        X = merged[feature_cols].fillna(0.0)
        y = merged["label"].astype(int)

        if HAS_LIGHTGBM:
            clf = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.05,
                random_state=42,
                verbose=-1,
            )
            backend_name = "lightgbm"
        else:
            clf = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.05,
                random_state=42,
            )
            backend_name = "sklearn"

        log.info("Fitting %s on %d labeled rows with %d features", backend_name, len(X), len(feature_cols))
        clf.fit(X, y)

        self._model = clf
        self._feature_columns = feature_cols
        self._model_version = MODEL_VERSION
        self._backend = backend_name
        self._is_fitted = True

        self.save()
        return self

    def save(self, path: str | Path | None = None) -> None:
        """Serialize model artifact to disk via joblib."""
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Cannot save an unfitted SupervisedScamModel")

        target_path = Path(path) if path else self.model_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        bundle = {
            "model": self._model,
            "version": self._model_version,
            "feature_columns": self._feature_columns,
            "backend": self._backend,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        if not HAS_JOBLIB:
            raise ImportError("joblib is required to save SupervisedScamModel")

        joblib.dump(bundle, target_path)
        log.info("Saved SupervisedScamModel artifact to %s", target_path)

    def load(self, path: str | Path | None = None) -> bool:
        """
        Deserialize model artifact from disk via joblib.

        Returns True if successfully loaded, False if model file is missing or invalid.
        """
        target_path = Path(path) if path else self.model_path
        if not target_path.exists() or not HAS_JOBLIB:
            self._is_fitted = False
            return False

        try:
            bundle = joblib.load(target_path)
            if not isinstance(bundle, dict) or "model" not in bundle:
                log.warning("Invalid model bundle in %s", target_path)
                self._is_fitted = False
                return False

            self._model = bundle["model"]
            self._model_version = bundle.get("version", MODEL_VERSION)
            self._feature_columns = list(bundle.get("feature_columns", []))
            self._backend = bundle.get("backend", "unknown")
            self._is_fitted = True
            log.info("Loaded SupervisedScamModel (v%s, backend=%s) from %s", self._model_version, self._backend, target_path)
            return True
        except Exception as exc:
            log.warning("Failed to load SupervisedScamModel from %s: %s", target_path, exc)
            self._is_fitted = False
            return False

    def predict_proba(self, feature_matrix: pd.DataFrame) -> list[float]:
        """
        Predict scam probability for a matrix of features.

        Returns a list of float probabilities [0.0, 1.0], or empty list if
        model is not fitted and cannot be loaded.
        """
        if not self._is_fitted:
            if not self.load():
                return []

        if feature_matrix.empty or self._model is None:
            return []

        # Ensure columns match expected training features
        X = feature_matrix.reindex(columns=self._feature_columns, fill_value=0.0).fillna(0.0)

        probas = self._model.predict_proba(X)[:, 1]
        return [float(p) for p in probas]

    def score_record(self, feature_vector: Any, record: Any = None) -> SupervisedResult:
        """Single-record inference helper."""
        if not self._is_fitted and not self.load():
            return SupervisedResult(model_available=False, scam_probability=None)

        df = assemble_feature_matrix([record or {}], [feature_vector])
        probas = self.predict_proba(df)

        if not probas:
            return SupervisedResult(model_available=False, scam_probability=None)

        return SupervisedResult(
            model_available=True,
            scam_probability=probas[0],
            model_version=self._model_version,
            backend=self._backend,
        )
