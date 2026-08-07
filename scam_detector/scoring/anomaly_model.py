"""
anomaly_model.py
----------------
Unsupervised anomaly detection layer that complements the deterministic
rules engine.

Bootstrapping strategy
----------------------
We do **not** yet have labeled fraud / not-fraud data.  This module therefore
uses unsupervised IsolationForest on the numeric feature vectors assembled
from Prompts 2–4 as a bootstrapping strategy until real labels exist from
human review (see Prompt 10).  Once a labeled corpus is available, this
layer can be replaced or calibrated against a supervised model without
changing the surrounding pipeline interface.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.ensemble import IsolationForest

# ---------------------------------------------------------------------------
# Feature-matrix assembly
# ---------------------------------------------------------------------------

# Numeric / boolean fields pulled from each FeatureVector slice.
# Column names are flat (domain__field) so the matrix is self-describing.
_TEXT_FIELDS: tuple[str, ...] = (
    "urgency_score",
    "caps_ratio",
    "exclamation_count",
    "has_repeated_punctuation",
    "genericity_score",
    "title_summary_alignment",
    "avg_sentence_length",
    "flesch_score",
    "artifact_count",
    "sensitive_info_requested",
    "boilerplate_similarity",
)

_COMPANY_FIELDS: tuple[str, ...] = (
    "is_suspect",
    "has_legal_suffix",
    "posting_count",
    "posting_date_span_days",
    "role_diversity_score",
    "typosquat_min_distance",
)

_URL_FIELDS: tuple[str, ...] = (
    "path_depth",
    "query_param_count",
    "is_https",
    "is_platform_internal",
    "is_url_shortener",
    "is_known_ats",
    "domain_entropy",  # url_entropy of the apply-link domain
    "domain_company_similarity",
    "tld_risk_score",
)

_STIPEND_FIELDS: tuple[str, ...] = (
    "is_outlier_high",
    "is_outlier_low",
    "pay_to_work_score",
    "missing_stipend_for_role",
    "amount_plausibility_score",
    "peer_zscore",  # stipend_zscore vs peer group
    "perk_consistency_ok",
    "hourly_inr",
)

_TEMPORAL_FIELDS: tuple[str, ...] = (
    "deadline_expired",
    "deadline_far_future",
    "posting_future_dated",
    "duration_implausible",
    "days_to_deadline",
    "duration_months",
    "deadline_urgency_score",
    "posting_burst_count",  # posting_burst_score count component
)

_STRUCTURAL_FIELDS: tuple[str, ...] = (
    "completeness_ratio",
    "skills_count",
    "skills_count_anomaly",
    "desc_to_title_ratio",
    "has_contact_in_body",
    "responsibilities_count",
    "openings_zscore",
    "field_completeness",
)

_SLICE_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("text", _TEXT_FIELDS),
    ("company", _COMPANY_FIELDS),
    ("url", _URL_FIELDS),
    ("stipend", _STIPEND_FIELDS),
    ("temporal", _TEMPORAL_FIELDS),
    ("structural", _STRUCTURAL_FIELDS),
)


def _to_mapping(obj: Any) -> Mapping[str, Any]:
    """Accept a pydantic model, dict, or duck-typed object with attributes."""
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if isinstance(obj, Mapping):
        return obj
    # Duck-type: read known slice attributes
    out: dict[str, Any] = {}
    for slice_name, _ in _SLICE_FIELDS:
        if hasattr(obj, slice_name):
            out[slice_name] = getattr(obj, slice_name)
    return out


def _coerce_numeric(value: Any) -> float:
    """Convert bools / numbers / None into a float suitable for IsolationForest."""
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float, np.integer, np.floating)):
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return 0.0
        return float(value)
    return 0.0


def _flatten_feature_vector(fv: Any) -> dict[str, float]:
    """Flatten one FeatureVector (or equivalent) into a flat numeric dict."""
    root = _to_mapping(fv)
    row: dict[str, float] = {}
    for slice_name, fields in _SLICE_FIELDS:
        slice_obj = root.get(slice_name, {})
        if isinstance(slice_obj, BaseModel):
            slice_map = slice_obj.model_dump()
        elif isinstance(slice_obj, Mapping):
            slice_map = slice_obj
        else:
            slice_map = {
                f: getattr(slice_obj, f, None) for f in fields
            } if slice_obj is not None else {}
        for field in fields:
            col = f"{slice_name}__{field}"
            row[col] = _coerce_numeric(slice_map.get(field))
    return row


def _record_id(record: Any, index: int) -> str:
    if isinstance(record, Mapping):
        for key in ("id", "_id", "record_id"):
            val = record.get(key)
            if val is not None and str(val).strip():
                return str(val)
    return f"row_{index}"


def assemble_feature_matrix(
    records: Sequence[Any],
    feature_vectors: Sequence[Any],
) -> pd.DataFrame:
    """
    Flatten pydantic feature-vector outputs into one row-per-record matrix.

    Parameters
    ----------
    records:
        Raw / remediated internship records (used for row index IDs).
        Length should match ``feature_vectors``; extras on either side are
        truncated to the shorter length.
    feature_vectors:
        Per-record :class:`~scam_detector.features.FeatureVector` instances
        (or dict / duck-typed equivalents) from Prompts 2–4.

    Returns
    -------
    pd.DataFrame
        Numeric matrix indexed by record id.  Boolean fields are encoded as
        0/1; missing optional floats (e.g. ``peer_zscore``) become 0.0 so
        IsolationForest can consume the matrix without imputation.
    """
    n = min(len(records), len(feature_vectors))
    rows: list[dict[str, float]] = []
    index: list[str] = []
    for i in range(n):
        index.append(_record_id(records[i], i))
        rows.append(_flatten_feature_vector(feature_vectors[i]))
    if not rows:
        # Empty frame with the expected column schema
        cols = [
            f"{slice_name}__{field}"
            for slice_name, fields in _SLICE_FIELDS
            for field in fields
        ]
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, index=index)


# ---------------------------------------------------------------------------
# AnomalyModel — IsolationForest on tabular features
# ---------------------------------------------------------------------------


class AnomalyModel:
    """
    Unsupervised IsolationForest anomaly detector over assembled feature matrices.

    Complements the rules engine when labeled fraud data is unavailable.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        contamination: float | str = "auto",
        random_state: int = 42,
        n_jobs: int = 1,
    ) -> None:
        self._model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self._fitted: bool = False
        self._feature_columns: list[str] = []
        self._train_mean: np.ndarray | None = None
        self._train_std: np.ndarray | None = None
        self._raw_score_min: float = 0.0
        self._raw_score_max: float = 1.0

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def feature_columns(self) -> list[str]:
        return list(self._feature_columns)

    def fit(self, feature_matrix: pd.DataFrame) -> AnomalyModel:
        """
        Train IsolationForest on the numeric feature vectors.

        Also stores per-feature mean/std of the training distribution for the
        lightweight ``explain`` approximation.
        """
        if feature_matrix.empty:
            raise ValueError("Cannot fit AnomalyModel on an empty feature matrix.")

        X = feature_matrix.astype(float).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        self._feature_columns = list(X.columns)
        values = X.to_numpy(dtype=np.float64)

        self._model.fit(values)
        self._train_mean = values.mean(axis=0)
        self._train_std = values.std(axis=0)
        # Avoid divide-by-zero for constant columns
        self._train_std = np.where(self._train_std < 1e-12, 1.0, self._train_std)

        raw = -self._model.decision_function(values)
        self._raw_score_min = float(raw.min())
        self._raw_score_max = float(raw.max())
        if abs(self._raw_score_max - self._raw_score_min) < 1e-12:
            # Degenerate case: all points equally anomalous — keep unit range
            self._raw_score_max = self._raw_score_min + 1.0

        self._fitted = True
        return self

    def _align(self, feature_matrix: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("AnomalyModel.score/explain called before fit().")
        aligned = feature_matrix.reindex(columns=self._feature_columns, fill_value=0.0)
        return (
            aligned.astype(float)
            .replace([np.inf, -np.inf], 0.0)
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )

    def _normalize_raw(self, raw: np.ndarray) -> np.ndarray:
        span = self._raw_score_max - self._raw_score_min
        normalized = (raw - self._raw_score_min) / span
        return np.clip(normalized, 0.0, 1.0)

    def score(self, feature_matrix: pd.DataFrame) -> np.ndarray:
        """
        Return anomaly scores normalized to ``[0, 1]`` (higher = more anomalous).

        Uses ``-decision_function`` from IsolationForest, min-max scaled against
        the training-set raw score range so scores are comparable across batches.
        """
        values = self._align(feature_matrix)
        if values.shape[0] == 0:
            return np.array([], dtype=np.float64)
        raw = -self._model.decision_function(values)
        return self._normalize_raw(raw)

    def explain(self, record_features: pd.Series | Mapping[str, Any]) -> list[tuple[str, float]]:
        """
        Approximate per-feature contribution via |z-score| vs training distribution.

        Sorted by magnitude (largest deviation first).  This is a lightweight
        stand-in for full SHAP-based explainability.

        TODO: upgrade to real SHAP values via the ``shap`` library once the
        pipeline is stable and the performance cost is justified.
        """
        if not self._fitted or self._train_mean is None or self._train_std is None:
            raise RuntimeError("AnomalyModel.explain called before fit().")

        if isinstance(record_features, pd.Series):
            series = record_features.reindex(self._feature_columns, fill_value=0.0)
        else:
            series = pd.Series(dict(record_features)).reindex(
                self._feature_columns, fill_value=0.0
            )

        values = (
            series.astype(float)
            .replace([np.inf, -np.inf], 0.0)
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
        z = (values - self._train_mean) / self._train_std
        contributions = [
            (col, float(abs(z_i)))
            for col, z_i in zip(self._feature_columns, z)
        ]
        contributions.sort(key=lambda item: item[1], reverse=True)
        return contributions


# ---------------------------------------------------------------------------
# TextAnomalyModel — Phase-2+ extension point (NOT needed for MVP)
# ---------------------------------------------------------------------------


class TextAnomalyModel:
    """
    Phase-2+ extension — **not needed for the MVP**.

    Interface-only stub for an embedding-space anomaly detector that would
    score postings by distance from the centroid of "normal" postings, using
    sentence embeddings produced by :class:`~scam_detector.features.duplicate_detection.DuplicateIndex`
    (Prompt 5).

    Kept here so the architecture has a clear extension point without
    over-building it now.
    """

    def __init__(self) -> None:
        self._centroid: np.ndarray | None = None
        self._fitted: bool = False

    def fit(self, embeddings: np.ndarray) -> TextAnomalyModel:
        """
        Fit on an (N, D) embedding matrix of presumed-normal postings.

        Phase-2+: compute and store the centroid of the normal cluster.
        """
        # Phase-2+ extension — not implemented for MVP
        raise NotImplementedError(
            "TextAnomalyModel is a Phase-2+ extension and is not implemented for MVP. "
            "Use AnomalyModel (IsolationForest on tabular features) instead."
        )

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Return anomaly scores in ``[0, 1]`` from distance to the normal centroid.

        Phase-2+: cosine / Euclidean distance from ``self._centroid``, normalized.
        """
        # Phase-2+ extension — not implemented for MVP
        raise NotImplementedError(
            "TextAnomalyModel is a Phase-2+ extension and is not implemented for MVP. "
            "Use AnomalyModel (IsolationForest on tabular features) instead."
        )

    def explain(self, embedding: np.ndarray) -> list[tuple[str, float]]:
        """
        Phase-2+: return a coarse explanation (e.g. distance-to-centroid only).
        """
        # Phase-2+ extension — not implemented for MVP
        raise NotImplementedError(
            "TextAnomalyModel is a Phase-2+ extension and is not implemented for MVP."
        )
