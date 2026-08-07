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
