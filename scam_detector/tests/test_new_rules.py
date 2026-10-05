"""
test_new_rules.py
------------------
Comprehensive unit tests for the newly added fraud detection rules:
  - UpfrontFeeAndPayToWorkRule
  - SuspiciousRecruiterContactRule
  - UrgencyAndPsychologicalPressureRule
  - Integration with RulesEngine noisy-OR aggregation
"""

from __future__ import annotations

import pytest

from scam_detector.config import Config
from scam_detector.scoring.rules_engine import (
    RuleFinding,
    RuleInput,
    RulesEngine,
    SuspiciousRecruiterContactRule,
    UpfrontFeeAndPayToWorkRule,
    UrgencyAndPsychologicalPressureRule,
)


# ---------------------------------------------------------------------------
# Test UpfrontFeeAndPayToWorkRule
# ---------------------------------------------------------------------------


class TestUpfrontFeeAndPayToWorkRule:
    def test_triggers_on_payment_required_flag(self) -> None:
        rule = UpfrontFeeAndPayToWorkRule()
        inp = RuleInput(payment_required=True)
        res = rule.evaluate(inp)
        assert res.triggered is True
        assert res.rule_id == "upfront_fee_and_pay_to_work"
        assert res.weight >= 0.85
        assert "upfront payment" in res.explanation

    def test_triggers_on_positive_registration_fee(self) -> None:
        rule = UpfrontFeeAndPayToWorkRule()
        inp = RuleInput(registration_fee=1500.0)
        res = rule.evaluate(inp)
        assert res.triggered is True
        assert "INR 1500.00" in res.explanation

    def test_triggers_on_fake_certificate_offer(self) -> None:
        rule = UpfrontFeeAndPayToWorkRule()
        inp = RuleInput(fake_certificate_offer=True)
        res = rule.evaluate(inp)
        assert res.triggered is True
        assert "certificate sales" in res.explanation

    def test_does_not_trigger_on_clean_listing(self) -> None:
        rule = UpfrontFeeAndPayToWorkRule()
        inp = RuleInput(payment_required=False, registration_fee=0.0, fake_certificate_offer=False)
        res = rule.evaluate(inp)
        assert res.triggered is False
        assert "No upfront fees" in res.explanation

    def test_custom_weight_applied(self) -> None:
        cfg = Config()
        cfg.rule_weights.upfront_fee_and_pay_to_work = 0.99
        rule = UpfrontFeeAndPayToWorkRule(cfg)
        res = rule.evaluate(RuleInput(payment_required=True))
        assert res.triggered is True
        assert res.weight == 0.99


# ---------------------------------------------------------------------------
# Test SuspiciousRecruiterContactRule
# ---------------------------------------------------------------------------


class TestSuspiciousRecruiterContactRule:
    def test_triggers_on_suspicious_email_domain(self) -> None:
        rule = SuspiciousRecruiterContactRule()
        inp = RuleInput(suspicious_email_domain=True)
        res = rule.evaluate(inp)
        assert res.triggered is True
        assert res.rule_id == "suspicious_recruiter_contact"
        assert "suspicious/unverified" in res.explanation

    def test_triggers_on_free_webmail_provider(self) -> None:
        rule = SuspiciousRecruiterContactRule()
        inp = RuleInput(recruiter_email_type="Free")
        res = rule.evaluate(inp)
        assert res.triggered is True
        assert "free webmail" in res.explanation

    def test_triggers_on_disposable_email_provider(self) -> None:
        rule = SuspiciousRecruiterContactRule()
        inp = RuleInput(recruiter_email_type="Disposable")
        res = rule.evaluate(inp)
        assert res.triggered is True
        assert "disposable webmail" in res.explanation

    def test_does_not_trigger_on_corporate_recruiter(self) -> None:
        rule = SuspiciousRecruiterContactRule()
        inp = RuleInput(recruiter_email_type="Corporate", suspicious_email_domain=False)
        res = rule.evaluate(inp)
        assert res.triggered is False
        assert "corporate-affiliated" in res.explanation


# ---------------------------------------------------------------------------
# Test UrgencyAndPsychologicalPressureRule
# ---------------------------------------------------------------------------


class TestUrgencyAndPsychologicalPressureRule:
    def test_triggers_on_high_urgency_ratio(self) -> None:
        rule = UrgencyAndPsychologicalPressureRule()
        inp = RuleInput(urgency_score=0.85)
        res = rule.evaluate(inp)
        assert res.triggered is True
        assert "urgency score" in res.explanation

    def test_triggers_on_scaled_urgency_score(self) -> None:
        rule = UrgencyAndPsychologicalPressureRule()
        # Scale 0-100 score: 80 / 100 = 0.80 > 0.65 threshold
        inp = RuleInput(urgency_score=80.0)
        res = rule.evaluate(inp)
        assert res.triggered is True
        assert "urgency score" in res.explanation

    def test_triggers_on_emotional_manipulation_score(self) -> None:
        rule = UrgencyAndPsychologicalPressureRule()
        inp = RuleInput(emotional_manipulation_score=0.75)
        res = rule.evaluate(inp)
        assert res.triggered is True
        assert "emotional manipulation" in res.explanation

    def test_triggers_on_phishing_language_score(self) -> None:
        rule = UrgencyAndPsychologicalPressureRule()
        inp = RuleInput(phishing_language_score=65.0)
        res = rule.evaluate(inp)
        assert res.triggered is True
        assert "phishing language" in res.explanation

    def test_does_not_trigger_on_moderate_neutral_posting(self) -> None:
        rule = UrgencyAndPsychologicalPressureRule()
        inp = RuleInput(
            urgency_score=0.20,
            emotional_manipulation_score=0.15,
            phishing_language_score=0.10,
        )
        res = rule.evaluate(inp)
        assert res.triggered is False
        assert "normal bounds" in res.explanation


# ---------------------------------------------------------------------------
# Test Full RulesEngine Integration with New Rules
# ---------------------------------------------------------------------------


class TestRulesEngineIntegration:
    def test_all_18_rules_registered_by_default(self) -> None:
        engine = RulesEngine()
        assert len(engine._rules) == 18
        rule_ids = [r.rule_id for r in engine._rules]
        assert "upfront_fee_and_pay_to_work" in rule_ids
        assert "suspicious_recruiter_contact" in rule_ids
        assert "urgency_psychological_pressure" in rule_ids
        assert "guaranteed_outcome_claim" in rule_ids
        assert "ngo_fundraising_stipend_network" in rule_ids
        assert "external_form_handoff" in rule_ids
        assert "training_program_disguised_as_internship" in rule_ids
        assert "zero_shot_semantic_scam_signal" in rule_ids
        assert "young_domain_age" in rule_ids
        assert "similar_to_confirmed_scam" in rule_ids

    def test_hard_reject_triggered_by_upfront_fee(self) -> None:
        engine = RulesEngine()
        inp = RuleInput(payment_required=True, registration_fee=500.0)
        res = engine.run(inp)
        assert res.is_hard_reject is True
        assert "upfront_fee_and_pay_to_work" in res.triggered_rule_ids
        assert res.combined_score >= 0.90

    def test_noisy_or_accumulation_across_multiple_rules(self) -> None:
        engine = RulesEngine()
        inp = RuleInput(
            suspicious_email_domain=True,
            urgency_score=0.85,
            is_platform_internal=False,
            domain_company_similarity=0.10,
        )
        res = engine.run(inp)
        assert res.combined_score > 0.60
        assert res.combined_score <= 1.0
        assert "suspicious_recruiter_contact" in res.triggered_rule_ids
        assert "urgency_psychological_pressure" in res.triggered_rule_ids
