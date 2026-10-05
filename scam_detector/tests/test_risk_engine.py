"""
test_risk_engine.py
-------------------
Capstone RiskEngine tests: hard-disqualifying override, low-confidence
gate, and clean high-confidence clear path.
"""

from __future__ import annotations

import pytest

from scam_detector.config import Config, BlendWeights, ConfidenceConfig, DecisionThresholds
from scam_detector.features import (
    FeatureVector,
    StructuralFeatures,
    CompanyFeatures,
)
from scam_detector.scoring.rules_engine import RuleFinding, RulesResult
from scam_detector.scoring.risk_engine import (
    RiskEngine,
    compute_confidence_score,
)
from scam_detector.scoring.explain import render_explanation, ScamScoreResult


def _finding(
    rule_id: str,
    *,
    weight: float,
    triggered: bool = True,
    description: str = "",
    explanation: str = "",
) -> RuleFinding:
    return RuleFinding(
        rule_id=rule_id,
        description=description or rule_id.replace("_", " "),
        weight=weight,
        triggered=triggered,
        explanation=explanation or f"{rule_id} fired",
    )


def _rules(
    *findings: RuleFinding,
    combined_score: float | None = None,
) -> RulesResult:
    triggered = [f for f in findings if f.triggered]
    if combined_score is None:
        if triggered:
            product = 1.0
            for f in triggered:
                product *= 1.0 - f.weight
            combined_score = 1.0 - product
        else:
            combined_score = 0.0
    return RulesResult(
        triggered=triggered,
        all_findings=list(findings),
        combined_score=round(combined_score, 4),
        is_hard_reject=any(f.weight >= 0.75 and f.triggered for f in findings),
        triggered_rule_ids=[f.rule_id for f in triggered],
        explanations=[f.explanation for f in triggered],
    )


class TestHardDisqualifyingOverride:
    def test_hard_rule_forces_block_despite_low_anomaly(self) -> None:
        engine = RiskEngine(
            Config(hard_disqualifying_decision="block")
        )
        rules = _rules(
            _finding(
                "hard_disqualifying_signals",
                weight=0.95,
                description="Hard disqualifying signals detected",
                explanation="Requests Aadhaar and upfront payment.",
            )
        )
        # Even with near-zero anomaly and mid confidence, must not clear
        result = engine.score_record(
            record={},
            rule_result=rules,
            anomaly_score=0.01,
            confidence_score=0.9,
        )
        assert result.decision in ("block", "review")
        assert result.decision == "block"
        assert result.hard_disqualifying_forced is True
        assert "hard_disqualifying_signals" in result.triggered_rules

    def test_hard_rule_can_be_configured_to_review(self) -> None:
        engine = RiskEngine(
            Config(hard_disqualifying_decision="review")
        )
        rules = _rules(
            _finding("hard_disqualifying_signals", weight=0.95)
        )
        result = engine.score_record(
            record={},
            rule_result=rules,
            anomaly_score=0.0,
            confidence_score=0.95,
        )
        assert result.decision == "review"
        assert result.hard_disqualifying_forced is True


class TestLowConfidenceGate:
    def test_low_confidence_must_not_be_clear(self) -> None:
        engine = RiskEngine(
            Config(
                confidence=ConfidenceConfig(low_below=0.4),
                decision_thresholds=DecisionThresholds(
                    clear_below=30.0,
                    block_at_or_above=70.0,
                ),
                blend_weights=BlendWeights(rules_weight=0.6, anomaly_weight=0.4),
            )
        )
        # No rules, tiny anomaly → raw score would be clear
        rules = _rules()
        result = engine.score_record(
            record={"flags": {"company_suspect": True}},
            rule_result=rules,
            anomaly_score=0.05,
            confidence_score=0.25,  # explicitly low
        )
        assert result.scam_score < 30.0
        assert result.decision != "clear"
        assert result.decision == "review"
        assert result.low_confidence_forced_review is True
        assert result.confidence_level == "low"


class TestCleanHighConfidenceClear:
    def test_clean_high_confidence_is_clear(self) -> None:
        engine = RiskEngine()
        rules = _rules()
        result = engine.score_record(
            record={},
            rule_result=rules,
            anomaly_score=0.05,
            confidence_score=0.92,
        )
        assert result.scam_score < 30.0
        assert result.decision == "clear"
        assert result.confidence_level == "high"
        assert result.hard_disqualifying_forced is False
        assert result.low_confidence_forced_review is False


class TestComputeConfidence:
    def test_suspect_flags_reduce_confidence(self) -> None:
        features = FeatureVector(
            structural=StructuralFeatures(field_completeness=1.0),
            company=CompanyFeatures(is_suspect=True),
        )
        record = {"flags": {"company_suspect": True, "degree_suspect_default": True}}
        score = compute_confidence_score(record, features)
        assert score < 0.5

    def test_complete_clean_record_is_high(self) -> None:
        features = FeatureVector(
            structural=StructuralFeatures(field_completeness=0.95),
        )
        score = compute_confidence_score({"flags": {}}, features)
        from scam_detector.features.text_features import _sbert_model
        if _sbert_model() is None:
            assert score >= 0.75
        else:
            assert score >= 0.9


class TestRenderExplanation:
    def test_report_includes_rules_features_and_caveat(self) -> None:
        engine = RiskEngine()
        rules = _rules(
            _finding(
                "cross_company_duplicate",
                weight=0.80,
                description="Cross-company near-duplicate text detected",
                explanation="Matches another company at 0.94 similarity.",
            ),
            _finding(
                "extreme_stipend_outlier",
                weight=0.45,
                description="Extreme stipend outlier (vs peer group)",
                explanation="Stipend z-score = 4.2 above peers.",
            ),
        )
        result = engine.score_record(
            record={},
            rule_result=rules,
            anomaly_score=0.7,
            confidence_score=0.2,
            feature_contributions=[
                ("stipend__peer_zscore", 4.2),
                ("text__urgency_score", 3.1),
                ("structural__openings_zscore", 2.8),
                ("url__domain_entropy", 2.1),
                ("temporal__posting_burst_count", 1.9),
                ("text__genericity_score", 1.2),
            ],
        )
        report = render_explanation(result)
        assert "cross_company_duplicate" in report
        assert "Matches another company" in report
        assert "stipend__peer_zscore" in report
        assert "CONFIDENCE CAVEAT" in report
        assert isinstance(result, ScamScoreResult)
        assert result.explanation_summary

    def test_render_explanation_with_shap_labels(self) -> None:
        from scam_detector.scoring.explain import render_explanation

        engine = RiskEngine()
        rules = _rules(_finding("typosquat_domain", weight=0.50))
        result = engine.score_record(
            record={},
            rule_result=rules,
            anomaly_score=0.45,
            confidence_score=0.90,
            feature_contributions=[("text__urgency_score", 0.3541), ("url__domain_entropy", 0.1234)],
            explanation_method="shap",
        )
        report = render_explanation(result)
        assert "Top contributing anomaly features (SHAP values):" in report
        assert "text__urgency_score: +0.3541 (SHAP)" in report

    def test_render_explanation_with_zscore_labels(self) -> None:
        from scam_detector.scoring.explain import render_explanation

        engine = RiskEngine()
        rules = _rules(_finding("typosquat_domain", weight=0.50))
        result = engine.score_record(
            record={},
            rule_result=rules,
            anomaly_score=0.45,
            confidence_score=0.90,
            feature_contributions=[("text__urgency_score", 3.54), ("url__domain_entropy", 1.23)],
            explanation_method="z_score_approximation",
        )
        report = render_explanation(result)
        assert "Top contributing anomaly features (|z-score| approx):" in report
        assert "text__urgency_score: 3.5400 (z-score)" in report


class TestBlendAndThresholds:
    def test_high_blended_score_blocks(self) -> None:
        engine = RiskEngine()
        rules = _rules(
            _finding("cross_company_duplicate", weight=0.80),
            _finding("typosquat_domain", weight=0.70),
        )
        result = engine.score_record(
            record={},
            rule_result=rules,
            anomaly_score=0.9,
            confidence_score=0.85,
        )
        assert result.scam_score >= 70.0
        assert result.decision == "block"
        assert result.hard_disqualifying_forced is False


class TestSupervisedBlendingAndCalibration:
    def test_supervised_score_blending_when_enabled(self) -> None:
        cfg = Config(
            blend_weights=BlendWeights(
                rules_weight=0.50,
                anomaly_weight=0.25,
                supervised_weight=0.25,
            )
        )
        engine = RiskEngine(cfg)
        rules = _rules(_finding("typosquat_domain", weight=0.60))

        # rules_score = 0.60, anomaly = 0.20, supervised = 0.80
        # Blend = 0.50*0.60 + 0.25*0.20 + 0.25*0.80 = 0.30 + 0.05 + 0.20 = 0.55
        result = engine.score_record(
            record={},
            rule_result=rules,
            anomaly_score=0.20,
            confidence_score=0.90,
            supervised_score=0.80,
        )
        assert result.supervised_score == 0.80
        assert abs(result.scam_score - 55.0) < 1.0

    def test_supervised_score_ignored_when_weight_zero(self) -> None:
        cfg = Config(
            blend_weights=BlendWeights(
                rules_weight=0.60,
                anomaly_weight=0.40,
                supervised_weight=0.0,
            )
        )
        engine = RiskEngine(cfg)
        rules = _rules(_finding("typosquat_domain", weight=0.50))

        # rules_score = 0.50, anomaly = 0.50 -> blended = 0.50 -> 50.0
        result = engine.score_record(
            record={},
            rule_result=rules,
            anomaly_score=0.50,
            confidence_score=0.90,
            supervised_score=0.99,  # Provided but weight is 0.0
        )
        assert result.supervised_score is None
        assert abs(result.scam_score - 50.0) < 1.0

    def test_calibrator_integration(self, tmp_path) -> None:
        from scam_detector.scoring.calibration import ScoreCalibrator

        raw_scores = [0.1 * i for i in range(10)] * 5
        labels = [0 if s < 0.5 else 1 for s in raw_scores]
        calibrator = ScoreCalibrator(min_samples=10).fit(raw_scores, labels)

        engine = RiskEngine(calibrator=calibrator)
        rules = _rules()

        result = engine.score_record(
            record={},
            rule_result=rules,
            anomaly_score=0.20,
            confidence_score=0.90,
        )
        assert 0.0 <= result.scam_score <= 100.0


class TestSourceConditionedConfidence:
    def test_unseen_source_fallback(self) -> None:
        cfg = Config(
            confidence=ConfidenceConfig(
                enable_source_conditioning=True,
                source_baseline_path="nonexistent_baselines_file.json",
            )
        )
        features = FeatureVector(
            structural=StructuralFeatures(field_completeness=0.50),
        )
        score_conditioned = compute_confidence_score(
            {"source": "new_unseen_source", "flags": {}},
            features,
            config=cfg,
        )

        cfg_disabled = Config(
            confidence=ConfidenceConfig(
                enable_source_conditioning=False,
            )
        )
        score_disabled = compute_confidence_score(
            {"source": "new_unseen_source", "flags": {}},
            features,
            config=cfg_disabled,
        )
        assert score_conditioned == score_disabled

    def test_existing_source_scaling(self, tmp_path) -> None:
        import json
        from scam_detector.scoring.risk_engine import clear_baselines_cache
        clear_baselines_cache()

        baseline_file = tmp_path / "test_source_baselines.json"
        baselines = {
            "unstop": {
                "mean_completeness": 0.50,
                "count": 100
            }
        }
        with open(baseline_file, "w", encoding="utf-8") as fh:
            json.dump(baselines, fh)

        cfg = Config(
            confidence=ConfidenceConfig(
                enable_source_conditioning=True,
                source_baseline_path=str(baseline_file),
                global_completeness_target=0.75,
            )
        )
        features = FeatureVector(
            structural=StructuralFeatures(field_completeness=0.50),
        )

        score_conditioned = compute_confidence_score(
            {"source": "unstop", "flags": {}},
            features,
            config=cfg,
        )

        cfg_disabled = Config(
            confidence=ConfidenceConfig(
                enable_source_conditioning=False,
            )
        )
        score_disabled = compute_confidence_score(
            {"source": "unstop", "flags": {}},
            features,
            config=cfg_disabled,
        )

        # Expected values depend on whether the optional sentence-transformers
        # dependency is installed: compute_confidence_score applies an extra
        # *0.80 penalty when it's missing (see its SBERT-availability check).
        # Hardcoding one environment's constants here made this test flake
        # depending on the machine it ran on — derive the multiplier instead.
        from scam_detector.features.text_features import _sbert_model
        sbert_penalty = 1.0 if _sbert_model() is not None else 0.80

        # completeness (0.50) * (global_target 0.75 / source mean 0.50) = 0.75
        expected_conditioned = round(0.75 * sbert_penalty, 4)
        # source-conditioning disabled → completeness passes through unscaled
        expected_disabled = round(0.50 * sbert_penalty, 4)

        assert score_conditioned > score_disabled
        assert abs(score_conditioned - expected_conditioned) < 1e-4
        assert abs(score_disabled - expected_disabled) < 1e-4

        clear_baselines_cache()

