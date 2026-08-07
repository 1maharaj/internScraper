"""
test_scoring.py
---------------
Smoke-tests for scam_detector.scoring and the top-level pipeline
"""

from scam_detector.scoring import (
    RiskLabel,
    RulesResult,
    RiskEngineResult,
    ScamScoreResult,
    apply_rules,
    apply_risk_engine,
    explain,
)
from scam_detector.features import FeatureVector
from scam_detector import ScamDetectorPipeline


# ── Rules engine ──────────────────────────────────────────────────────────────

def test_apply_rules_returns_rules_result() -> None:
    fv = FeatureVector()
    result = apply_rules(fv)
    assert isinstance(result, RulesResult)


def test_rules_result_no_hard_reject_by_default() -> None:
    fv = FeatureVector()
    result = apply_rules(fv)
    assert result.is_hard_reject is False


def test_rules_result_combined_score_in_range() -> None:
    fv = FeatureVector()
    result = apply_rules(fv)
    assert 0.0 <= result.combined_score <= 1.0


# ── Risk engine ───────────────────────────────────────────────────────────────

def test_apply_risk_engine_returns_risk_engine_result() -> None:
    fv = FeatureVector()
    result = apply_risk_engine(fv)
    assert isinstance(result, RiskEngineResult)


def test_risk_engine_no_model_is_neutral() -> None:
    fv = FeatureVector()
    result = apply_risk_engine(fv)
    assert result.model_available is False
    assert result.score == 0.5


# ── Explain ───────────────────────────────────────────────────────────────────

def test_explain_returns_scam_score_result() -> None:
    fv = FeatureVector()
    rules = apply_rules(fv)
    risk = apply_risk_engine(fv)
    result = explain(rules, risk, fv)
    assert isinstance(result, ScamScoreResult)


def test_explain_label_is_risk_label() -> None:
    fv = FeatureVector()
    result = explain(apply_rules(fv), apply_risk_engine(fv), fv)
    assert isinstance(result.label, RiskLabel)


def test_explain_result_has_decision_and_scam_score() -> None:
    fv = FeatureVector()
    result = explain(apply_rules(fv), apply_risk_engine(fv), fv)
    assert result.decision in ("clear", "review", "block")
    assert 0.0 <= result.scam_score <= 100.0
    assert 0.0 <= result.confidence <= 1.0


# ── Full pipeline ─────────────────────────────────────────────────────────────

def test_pipeline_run_returns_scam_score_result() -> None:
    raw = {"name": "Backend Intern", "company": "TechCo", "apply_link": "https://techco.com/jobs"}
    result = ScamDetectorPipeline().run(raw)
    assert isinstance(result, ScamScoreResult)


def test_pipeline_run_empty_input() -> None:
    result = ScamDetectorPipeline().run({})
    assert isinstance(result, ScamScoreResult)
    assert 0.0 <= result.final_score <= 1.0
