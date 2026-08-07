"""
test_anomaly_model.py
---------------------
Tests for the unsupervised IsolationForest anomaly layer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scam_detector.features import (
    FeatureVector,
    TextFeatureVector,
    CompanyFeatures,
    UrlFeatures,
    StipendFeatures,
    TemporalFeatures,
    StructuralFeatures,
)
from scam_detector.scoring.anomaly_model import (
    AnomalyModel,
    TextAnomalyModel,
    assemble_feature_matrix,
)


def _normal_vector(**overrides: object) -> FeatureVector:
    """A typical, low-risk feature vector."""
    text = TextFeatureVector(
        urgency_score=0.05,
        caps_ratio=0.02,
        exclamation_count=0,
        genericity_score=0.2,
        title_summary_alignment=0.85,
        avg_sentence_length=18.0,
        flesch_score=60.0,
        artifact_count=0,
        sensitive_info_requested=False,
        boilerplate_similarity=0.1,
    )
    company = CompanyFeatures(
        is_suspect=False,
        has_legal_suffix=True,
        posting_count=3,
        posting_date_span_days=14.0,
        role_diversity_score=0.5,
        typosquat_min_distance=1.0,
    )
    url = UrlFeatures(
        path_depth=2,
        query_param_count=0,
        is_https=True,
        is_platform_internal=True,
        domain_entropy=2.5,
        domain_company_similarity=0.9,
        tld_risk_score=0.0,
    )
    stipend = StipendFeatures(
        peer_zscore=0.1,
        amount_plausibility_score=0.9,
        perk_consistency_ok=True,
        hourly_inr=80.0,
    )
    temporal = TemporalFeatures(
        posting_burst_count=1,
        deadline_urgency_score=0.2,
        days_to_deadline=30.0,
    )
    structural = StructuralFeatures(
        openings_zscore=0.0,
        field_completeness=0.8,
        completeness_ratio=0.8,
        skills_count=5,
    )
    fv = FeatureVector(
        text=text,
        company=company,
        url=url,
        stipend=stipend,
        temporal=temporal,
        structural=structural,
    )
    # Apply nested overrides like text__urgency_score via model_copy if needed
    if overrides:
        data = fv.model_dump()
        for key, value in overrides.items():
            if "__" in key:
                slice_name, field = key.split("__", 1)
                data[slice_name][field] = value
            else:
                data[key] = value
        fv = FeatureVector.model_validate(data)
    return fv


def _outlier_urgent() -> FeatureVector:
    return _normal_vector(
        text__urgency_score=1.0,
        text__caps_ratio=0.95,
        text__exclamation_count=40,
        text__genericity_score=0.99,
        text__title_summary_alignment=0.05,
        text__sensitive_info_requested=True,
        stipend__peer_zscore=8.5,
        structural__openings_zscore=6.0,
        temporal__posting_burst_count=50,
        url__domain_entropy=4.8,
        url__tld_risk_score=1.0,
        url__domain_company_similarity=0.0,
    )


def _outlier_mass_openings() -> FeatureVector:
    return _normal_vector(
        structural__openings_zscore=9.0,
        text__genericity_score=0.98,
        temporal__posting_burst_count=80,
        stipend__peer_zscore=-5.0,
        company__is_suspect=True,
        company__posting_count=200,
        url__is_url_shortener=True,
        url__domain_entropy=4.5,
    )


def _outlier_stipend_scam() -> FeatureVector:
    return _normal_vector(
        stipend__peer_zscore=12.0,
        stipend__pay_to_work_score=1.0,
        text__urgency_score=0.95,
        text__boilerplate_similarity=0.99,
        structural__openings_zscore=5.5,
        url__is_https=False,
        url__tld_risk_score=0.9,
    )


class TestAssembleFeatureMatrix:
    def test_one_row_per_record(self) -> None:
        records = [{"id": "a"}, {"id": "b"}]
        vectors = [_normal_vector(), _outlier_urgent()]
        matrix = assemble_feature_matrix(records, vectors)
        assert isinstance(matrix, pd.DataFrame)
        assert len(matrix) == 2
        assert list(matrix.index) == ["a", "b"]

    def test_contains_expected_columns(self) -> None:
        matrix = assemble_feature_matrix(
            [{"id": "x"}],
            [_normal_vector()],
        )
        for col in (
            "text__title_summary_alignment",
            "text__urgency_score",
            "stipend__peer_zscore",
            "structural__openings_zscore",
            "temporal__posting_burst_count",
            "url__domain_entropy",
        ):
            assert col in matrix.columns

    def test_all_numeric(self) -> None:
        matrix = assemble_feature_matrix([{}], [_normal_vector()])
        assert matrix.select_dtypes(include=[np.number]).shape[1] == matrix.shape[1]

    def test_empty_inputs(self) -> None:
        matrix = assemble_feature_matrix([], [])
        assert matrix.empty


class TestAnomalyModel:
    def _synthetic_corpus(self) -> tuple[list[dict], list[FeatureVector]]:
        # 12 normal + 3 obvious outliers
        normals = [_normal_vector() for _ in range(12)]
        # Mild jitter so IsolationForest sees a real cloud, not identical points
        rng = np.random.default_rng(0)
        jittered: list[FeatureVector] = []
        for fv in normals:
            data = fv.model_dump()
            data["text"]["urgency_score"] = float(
                np.clip(0.05 + rng.normal(0, 0.02), 0, 1)
            )
            data["text"]["caps_ratio"] = float(
                np.clip(0.02 + rng.normal(0, 0.01), 0, 1)
            )
            data["stipend"]["peer_zscore"] = float(rng.normal(0.0, 0.3))
            data["structural"]["openings_zscore"] = float(rng.normal(0.0, 0.4))
            data["url"]["domain_entropy"] = float(2.5 + rng.normal(0, 0.15))
            data["temporal"]["posting_burst_count"] = int(
                max(1, round(1 + rng.normal(0, 0.5)))
            )
            jittered.append(FeatureVector.model_validate(data))

        outliers = [
            _outlier_urgent(),
            _outlier_mass_openings(),
            _outlier_stipend_scam(),
        ]
        vectors = jittered + outliers
        records = [{"id": f"n{i}"} for i in range(12)] + [
            {"id": "out_urgent"},
            {"id": "out_mass"},
            {"id": "out_stipend"},
        ]
        return records, vectors

    def test_outliers_score_higher_than_normals(self) -> None:
        records, vectors = self._synthetic_corpus()
        matrix = assemble_feature_matrix(records, vectors)
        model = AnomalyModel(random_state=42, contamination=0.2)
        model.fit(matrix)
        scores = model.score(matrix)

        assert scores.shape == (len(matrix),)
        assert np.all(scores >= 0.0) and np.all(scores <= 1.0)

        normal_scores = scores[:12]
        outlier_scores = scores[12:]

        # Every synthetic outlier should beat the normal mean
        assert float(outlier_scores.min()) > float(normal_scores.mean())
        # And collectively score higher than the normal max
        assert float(outlier_scores.mean()) > float(normal_scores.max())

    def test_explain_sorted_by_magnitude(self) -> None:
        records, vectors = self._synthetic_corpus()
        matrix = assemble_feature_matrix(records, vectors)
        model = AnomalyModel(random_state=42).fit(matrix)

        contribs = model.explain(matrix.iloc[-1])
        assert isinstance(contribs, list)
        assert all(isinstance(name, str) and isinstance(val, float) for name, val in contribs)
        mags = [v for _, v in contribs]
        assert mags == sorted(mags, reverse=True)

    def test_fit_empty_raises(self) -> None:
        model = AnomalyModel()
        with pytest.raises(ValueError, match="empty"):
            model.fit(pd.DataFrame())

    def test_score_before_fit_raises(self) -> None:
        model = AnomalyModel()
        with pytest.raises(RuntimeError, match="before fit"):
            model.score(pd.DataFrame({"a": [1.0]}))


class TestTextAnomalyModelStub:
    def test_fit_raises_not_implemented(self) -> None:
        stub = TextAnomalyModel()
        with pytest.raises(NotImplementedError, match="Phase-2"):
            stub.fit(np.zeros((5, 8)))

    def test_score_raises_not_implemented(self) -> None:
        stub = TextAnomalyModel()
        with pytest.raises(NotImplementedError, match="Phase-2"):
            stub.score(np.zeros((2, 8)))
