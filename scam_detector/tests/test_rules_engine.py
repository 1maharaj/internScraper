"""
test_rules_engine.py
--------------------
Comprehensive tests for scam_detector.scoring.rules_engine.

Coverage
--------
Each rule is tested individually with:
  - A synthetic RuleInput that SHOULD trigger the rule
  - A synthetic RuleInput that should NOT trigger the rule
  - Edge-case inputs (boundary values, None fields, neutral defaults)

RulesEngine integration tests verify:
  - Noisy-OR score formula correctness
  - is_hard_reject flag set/unset correctly
  - Custom config weights flow through to RuleFinding.weight
  - Empty input produces zero score and no hard reject
  - All 7 rules included in all_findings even when not triggered

Weight-configurability tests verify that changing a weight in Config
changes the returned RuleFinding.weight without altering trigger logic.

Structure
---------
One test class per rule, then TestRulesEngine for integration,
TestApplyRules for the legacy shim, and TestNoisyOrFormula for the
score-combination math.
"""

from __future__ import annotations

import math

import pytest

from scam_detector.config import Config, RuleWeights, RuleThresholds
from scam_detector.scoring.rules_engine import (
    RuleInput,
    RuleFinding,
    RulesResult,
    RulesEngine,
    HardDisqualifyingSignalsRule,
    GuaranteedOutcomeClaimRule,
    ExternalFormHandoffRule,
    TrainingProgramDisguisedAsInternshipRule,
    ZeroShotSemanticScamRule,
    YoungDomainAgeRule,
    SimilarToConfirmedScamRule,
    NgoFundraisingStipendNetworkRule,
    StipendPerkContradictionRule,
    CrossCompanyDuplicateRule,
    ExtremeStipendOutlierRule,
    UnverifiableCompanyRule,
    TyposquatDomainRule,
    MassOpeningsVagueRoleRule,
    apply_rules,
    _default_rules,
    _HARD_REJECT_WEIGHT_THRESHOLD,
)
from scam_detector.features import FeatureVector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean() -> RuleInput:
    """All-safe default input — no rule should trigger."""
    return RuleInput()


def _cfg(**rule_weight_overrides) -> Config:
    """Build a Config with specific rule weights overridden."""
    return Config(rule_weights=RuleWeights(**rule_weight_overrides))


def _cfg_thresholds(**threshold_overrides) -> Config:
    """Build a Config with specific rule thresholds overridden."""
    return Config(rule_thresholds=RuleThresholds(**threshold_overrides))



# ===========================================================================
# TestHardDisqualifyingSignalsRule
# ===========================================================================

class TestHardDisqualifyingSignalsRule:

    def setup_method(self) -> None:
        self.rule = HardDisqualifyingSignalsRule()

    # ── Trigger ───────────────────────────────────────────────────────────

    def test_triggers_when_sensitive_info_requested_true(self) -> None:
        inp = RuleInput(sensitive_info_requested=True)
        finding = self.rule.evaluate(inp)
        assert finding.triggered is True

    def test_triggered_rule_id_correct(self) -> None:
        finding = self.rule.evaluate(RuleInput(sensitive_info_requested=True))
        assert finding.rule_id == "hard_disqualifying_signals"

    def test_triggered_weight_matches_config_default(self) -> None:
        finding = self.rule.evaluate(RuleInput(sensitive_info_requested=True))
        assert finding.weight == pytest.approx(0.95)

    def test_triggered_explanation_mentions_payment_or_id(self) -> None:
        finding = self.rule.evaluate(RuleInput(sensitive_info_requested=True))
        explanation_lc = finding.explanation.lower()
        assert any(kw in explanation_lc for kw in ("payment", "deposit", "aadhaar", "bank"))

    # ── No-trigger ────────────────────────────────────────────────────────

    def test_does_not_trigger_on_clean_input(self) -> None:
        finding = self.rule.evaluate(_clean())
        assert finding.triggered is False

    def test_does_not_trigger_when_false(self) -> None:
        finding = self.rule.evaluate(RuleInput(sensitive_info_requested=False))
        assert finding.triggered is False

    def test_non_triggered_weight_still_set(self) -> None:
        finding = self.rule.evaluate(_clean())
        assert finding.weight == pytest.approx(0.95)

    def test_non_triggered_explanation_non_empty(self) -> None:
        assert self.rule.evaluate(_clean()).explanation != ""

    # ── Weight configurability ────────────────────────────────────────────

    def test_custom_weight_reflected_in_finding(self) -> None:
        rule = HardDisqualifyingSignalsRule(config=_cfg(hard_disqualifying_signals=0.50))
        finding = rule.evaluate(RuleInput(sensitive_info_requested=True))
        assert finding.weight == pytest.approx(0.50)
        assert finding.triggered is True

    def test_weight_change_does_not_affect_trigger_logic(self) -> None:
        rule_low = HardDisqualifyingSignalsRule(config=_cfg(hard_disqualifying_signals=0.10))
        rule_high = HardDisqualifyingSignalsRule(config=_cfg(hard_disqualifying_signals=0.99))
        for inp in (RuleInput(sensitive_info_requested=True),
                    RuleInput(sensitive_info_requested=False)):
            assert rule_low.evaluate(inp).triggered == rule_high.evaluate(inp).triggered

    # ── Return type ───────────────────────────────────────────────────────

    def test_returns_rule_finding_instance(self) -> None:
        assert isinstance(self.rule.evaluate(_clean()), RuleFinding)


# ===========================================================================
# TestGuaranteedOutcomeClaimRule
# ===========================================================================

class TestGuaranteedOutcomeClaimRule:

    def setup_method(self) -> None:
        self.rule = GuaranteedOutcomeClaimRule()

    # ── Trigger ───────────────────────────────────────────────────────────

    def test_triggers_when_guaranteed_outcome_claim_true(self) -> None:
        inp = RuleInput(guaranteed_outcome_claim=True)
        finding = self.rule.evaluate(inp)
        assert finding.triggered is True

    def test_triggered_rule_id_correct(self) -> None:
        finding = self.rule.evaluate(RuleInput(guaranteed_outcome_claim=True))
        assert finding.rule_id == "guaranteed_outcome_claim"

    def test_triggered_weight_matches_config_default(self) -> None:
        finding = self.rule.evaluate(RuleInput(guaranteed_outcome_claim=True))
        assert finding.weight == pytest.approx(0.85)

    def test_triggered_explanation_mentions_guarantee_or_interview(self) -> None:
        finding = self.rule.evaluate(RuleInput(guaranteed_outcome_claim=True))
        explanation_lc = finding.explanation.lower()
        assert any(kw in explanation_lc for kw in ("guarantee", "interview", "handoff"))

    # ── No-trigger ────────────────────────────────────────────────────────

    def test_does_not_trigger_on_clean_input(self) -> None:
        finding = self.rule.evaluate(_clean())
        assert finding.triggered is False

    def test_does_not_trigger_when_false(self) -> None:
        finding = self.rule.evaluate(RuleInput(guaranteed_outcome_claim=False))
        assert finding.triggered is False

    def test_non_triggered_weight_still_set(self) -> None:
        finding = self.rule.evaluate(_clean())
        assert finding.weight == pytest.approx(0.85)

    def test_non_triggered_explanation_non_empty(self) -> None:
        assert self.rule.evaluate(_clean()).explanation != ""

    # ── Weight configurability ────────────────────────────────────────────

    def test_custom_weight_reflected_in_finding(self) -> None:
        rule = GuaranteedOutcomeClaimRule(config=_cfg(guaranteed_outcome_claim=0.50))
        finding = rule.evaluate(RuleInput(guaranteed_outcome_claim=True))
        assert finding.weight == pytest.approx(0.50)
        assert finding.triggered is True

    def test_weight_change_does_not_affect_trigger_logic(self) -> None:
        rule_low = GuaranteedOutcomeClaimRule(config=_cfg(guaranteed_outcome_claim=0.10))
        rule_high = GuaranteedOutcomeClaimRule(config=_cfg(guaranteed_outcome_claim=0.99))
        for inp in (RuleInput(guaranteed_outcome_claim=True),
                    RuleInput(guaranteed_outcome_claim=False)):
            assert rule_low.evaluate(inp).triggered == rule_high.evaluate(inp).triggered

    # ── Return type ───────────────────────────────────────────────────────

    def test_returns_rule_finding_instance(self) -> None:
        assert isinstance(self.rule.evaluate(_clean()), RuleFinding)


# ===========================================================================
# TestExternalFormHandoffRule
# ===========================================================================

class TestExternalFormHandoffRule:

    def setup_method(self) -> None:
        self.rule = ExternalFormHandoffRule()

    # ── Trigger: both signals (escalated weight) ────────────────────────────

    def test_triggers_when_both_form_and_phone_present(self) -> None:
        inp = RuleInput(external_form_detected=True, personal_contact_handoff_detected=True)
        finding = self.rule.evaluate(inp)
        assert finding.triggered is True

    def test_both_present_escalates_weight_above_base(self) -> None:
        finding = self.rule.evaluate(
            RuleInput(external_form_detected=True, personal_contact_handoff_detected=True)
        )
        assert finding.weight == pytest.approx(0.85)

    def test_escalated_weight_capped_at_one(self) -> None:
        rule = ExternalFormHandoffRule(config=_cfg(external_form_handoff=0.90))
        finding = rule.evaluate(
            RuleInput(external_form_detected=True, personal_contact_handoff_detected=True)
        )
        assert finding.weight == pytest.approx(1.0)

    # ── Trigger: single signal (base weight) ────────────────────────────────

    def test_triggers_on_form_alone(self) -> None:
        finding = self.rule.evaluate(RuleInput(external_form_detected=True))
        assert finding.triggered is True
        assert finding.weight == pytest.approx(0.60)

    def test_triggers_on_phone_alone(self) -> None:
        finding = self.rule.evaluate(RuleInput(personal_contact_handoff_detected=True))
        assert finding.triggered is True
        assert finding.weight == pytest.approx(0.60)

    def test_triggered_rule_id_correct(self) -> None:
        finding = self.rule.evaluate(RuleInput(external_form_detected=True))
        assert finding.rule_id == "external_form_handoff"

    # ── No-trigger ────────────────────────────────────────────────────────

    def test_does_not_trigger_on_clean_input(self) -> None:
        finding = self.rule.evaluate(_clean())
        assert finding.triggered is False

    def test_non_triggered_weight_still_set(self) -> None:
        finding = self.rule.evaluate(_clean())
        assert finding.weight == pytest.approx(0.60)

    def test_non_triggered_explanation_non_empty(self) -> None:
        assert self.rule.evaluate(_clean()).explanation != ""

    # ── Weight configurability ────────────────────────────────────────────

    def test_custom_base_weight_reflected(self) -> None:
        rule = ExternalFormHandoffRule(config=_cfg(external_form_handoff=0.40))
        finding = rule.evaluate(RuleInput(external_form_detected=True))
        assert finding.weight == pytest.approx(0.40)

    # ── Return type ───────────────────────────────────────────────────────

    def test_returns_rule_finding_instance(self) -> None:
        assert isinstance(self.rule.evaluate(_clean()), RuleFinding)


# ===========================================================================
# TestTrainingProgramDisguisedAsInternshipRule
# ===========================================================================

class TestTrainingProgramDisguisedAsInternshipRule:

    def setup_method(self) -> None:
        self.rule = TrainingProgramDisguisedAsInternshipRule()

    # ── Trigger ───────────────────────────────────────────────────────────

    def test_triggers_when_true(self) -> None:
        inp = RuleInput(training_program_disguised_as_internship=True)
        finding = self.rule.evaluate(inp)
        assert finding.triggered is True

    def test_triggered_rule_id_correct(self) -> None:
        finding = self.rule.evaluate(RuleInput(training_program_disguised_as_internship=True))
        assert finding.rule_id == "training_program_disguised_as_internship"

    def test_triggered_weight_matches_config_default(self) -> None:
        finding = self.rule.evaluate(RuleInput(training_program_disguised_as_internship=True))
        assert finding.weight == pytest.approx(0.65)

    # ── No-trigger ────────────────────────────────────────────────────────

    def test_does_not_trigger_on_clean_input(self) -> None:
        finding = self.rule.evaluate(_clean())
        assert finding.triggered is False

    def test_non_triggered_weight_still_set(self) -> None:
        finding = self.rule.evaluate(_clean())
        assert finding.weight == pytest.approx(0.65)

    def test_non_triggered_explanation_non_empty(self) -> None:
        assert self.rule.evaluate(_clean()).explanation != ""

    # ── Weight configurability ────────────────────────────────────────────

    def test_custom_weight_reflected_in_finding(self) -> None:
        rule = TrainingProgramDisguisedAsInternshipRule(
            config=_cfg(training_program_disguised_as_internship=0.30)
        )
        finding = rule.evaluate(RuleInput(training_program_disguised_as_internship=True))
        assert finding.weight == pytest.approx(0.30)
        assert finding.triggered is True

    # ── Return type ───────────────────────────────────────────────────────

    def test_returns_rule_finding_instance(self) -> None:
        assert isinstance(self.rule.evaluate(_clean()), RuleFinding)


# ===========================================================================
# TestZeroShotSemanticScamRule
# ===========================================================================

class TestZeroShotSemanticScamRule:

    def setup_method(self) -> None:
        self.rule = ZeroShotSemanticScamRule()

    # ── Trigger ───────────────────────────────────────────────────────────

    def test_triggers_above_threshold(self) -> None:
        inp = RuleInput(
            zero_shot_scam_category="sells a paid training or certification course disguised as a job",
            zero_shot_scam_confidence=0.75,
        )
        finding = self.rule.evaluate(inp)
        assert finding.triggered is True

    def test_triggered_rule_id_correct(self) -> None:
        finding = self.rule.evaluate(
            RuleInput(zero_shot_scam_category="some category", zero_shot_scam_confidence=0.80)
        )
        assert finding.rule_id == "zero_shot_semantic_scam_signal"

    def test_triggered_weight_matches_config_default(self) -> None:
        finding = self.rule.evaluate(
            RuleInput(zero_shot_scam_category="some category", zero_shot_scam_confidence=0.80)
        )
        assert finding.weight == pytest.approx(0.55)

    def test_explanation_includes_category_and_confidence(self) -> None:
        finding = self.rule.evaluate(
            RuleInput(zero_shot_scam_category="promises a guaranteed job", zero_shot_scam_confidence=0.85)
        )
        assert "promises a guaranteed job" in finding.explanation
        assert "85%" in finding.explanation

    # ── No-trigger ────────────────────────────────────────────────────────

    def test_does_not_trigger_when_category_none(self) -> None:
        finding = self.rule.evaluate(_clean())
        assert finding.triggered is False

    def test_does_not_trigger_below_threshold(self) -> None:
        inp = RuleInput(zero_shot_scam_category="some category", zero_shot_scam_confidence=0.30)
        finding = self.rule.evaluate(inp)
        assert finding.triggered is False

    def test_does_not_trigger_exactly_at_threshold_boundary_minus_epsilon(self) -> None:
        inp = RuleInput(zero_shot_scam_category="some category", zero_shot_scam_confidence=0.59)
        finding = self.rule.evaluate(inp)
        assert finding.triggered is False

    def test_triggers_at_exact_threshold(self) -> None:
        inp = RuleInput(zero_shot_scam_category="some category", zero_shot_scam_confidence=0.60)
        finding = self.rule.evaluate(inp)
        assert finding.triggered is True

    def test_non_triggered_weight_still_set(self) -> None:
        finding = self.rule.evaluate(_clean())
        assert finding.weight == pytest.approx(0.55)

    # ── Weight configurability ────────────────────────────────────────────

    def test_custom_weight_reflected_in_finding(self) -> None:
        rule = ZeroShotSemanticScamRule(config=_cfg(zero_shot_semantic_scam_signal=0.20))
        finding = rule.evaluate(
            RuleInput(zero_shot_scam_category="some category", zero_shot_scam_confidence=0.80)
        )
        assert finding.weight == pytest.approx(0.20)
        assert finding.triggered is True

    # ── Return type ───────────────────────────────────────────────────────

    def test_returns_rule_finding_instance(self) -> None:
        assert isinstance(self.rule.evaluate(_clean()), RuleFinding)


# ===========================================================================
# TestYoungDomainAgeRule
# ===========================================================================

class TestYoungDomainAgeRule:

    def setup_method(self) -> None:
        self.rule = YoungDomainAgeRule()

    # ── Trigger ───────────────────────────────────────────────────────────

    def test_triggers_when_domain_younger_than_threshold(self) -> None:
        inp = RuleInput(domain_age_days=10)
        finding = self.rule.evaluate(inp)
        assert finding.triggered is True

    def test_triggered_rule_id_correct(self) -> None:
        finding = self.rule.evaluate(RuleInput(domain_age_days=5))
        assert finding.rule_id == "young_domain_age"

    def test_triggered_weight_matches_config_default(self) -> None:
        finding = self.rule.evaluate(RuleInput(domain_age_days=5))
        assert finding.weight == pytest.approx(0.55)

    def test_explanation_includes_days(self) -> None:
        finding = self.rule.evaluate(RuleInput(domain_age_days=7))
        assert "7" in finding.explanation

    # ── No-trigger ────────────────────────────────────────────────────────

    def test_does_not_trigger_when_none(self) -> None:
        # None = unknown/not-applicable (platform/ATS link) — must never trigger
        finding = self.rule.evaluate(_clean())
        assert finding.triggered is False

    def test_does_not_trigger_when_old_domain(self) -> None:
        finding = self.rule.evaluate(RuleInput(domain_age_days=3650))
        assert finding.triggered is False

    def test_does_not_trigger_at_exact_threshold(self) -> None:
        # threshold default is 90 — age must be strictly LESS than threshold
        finding = self.rule.evaluate(RuleInput(domain_age_days=90))
        assert finding.triggered is False

    def test_non_triggered_weight_still_set(self) -> None:
        finding = self.rule.evaluate(_clean())
        assert finding.weight == pytest.approx(0.55)

    def test_none_explanation_mentions_unknown(self) -> None:
        explanation = self.rule.evaluate(_clean()).explanation.lower()
        assert "unknown" in explanation or "not applicable" in explanation

    # ── Weight configurability ────────────────────────────────────────────

    def test_custom_weight_reflected_in_finding(self) -> None:
        rule = YoungDomainAgeRule(config=_cfg(young_domain_age=0.30))
        finding = rule.evaluate(RuleInput(domain_age_days=5))
        assert finding.weight == pytest.approx(0.30)
        assert finding.triggered is True

    # ── Threshold configurability ────────────────────────────────────────────

    def test_custom_threshold_reflected(self) -> None:
        rule = YoungDomainAgeRule(config=_cfg_thresholds(young_domain_age_days_threshold=365))
        finding = rule.evaluate(RuleInput(domain_age_days=200))
        assert finding.triggered is True

    # ── Return type ───────────────────────────────────────────────────────

    def test_returns_rule_finding_instance(self) -> None:
        assert isinstance(self.rule.evaluate(_clean()), RuleFinding)


# ===========================================================================
# TestSimilarToConfirmedScamRule
# ===========================================================================

class TestSimilarToConfirmedScamRule:

    def setup_method(self) -> None:
        self.rule = SimilarToConfirmedScamRule()

    # ── Trigger ───────────────────────────────────────────────────────────

    def test_triggers_above_threshold(self) -> None:
        finding = self.rule.evaluate(RuleInput(scam_corpus_similarity=0.90))
        assert finding.triggered is True

    def test_triggers_at_exact_threshold(self) -> None:
        # default threshold is 0.85 - must trigger AT the threshold, not just above
        finding = self.rule.evaluate(RuleInput(scam_corpus_similarity=0.85))
        assert finding.triggered is True

    def test_triggered_rule_id_correct(self) -> None:
        finding = self.rule.evaluate(RuleInput(scam_corpus_similarity=0.95))
        assert finding.rule_id == "similar_to_confirmed_scam"

    def test_triggered_weight_matches_config_default(self) -> None:
        finding = self.rule.evaluate(RuleInput(scam_corpus_similarity=0.95))
        assert finding.weight == pytest.approx(0.70)

    def test_weight_below_hard_reject_threshold(self) -> None:
        # Deliberate design choice: strong but not an automatic hard reject
        assert self.rule.evaluate(RuleInput(scam_corpus_similarity=0.95)).weight < 0.75

    def test_explanation_mentions_similarity_value(self) -> None:
        finding = self.rule.evaluate(RuleInput(scam_corpus_similarity=0.91))
        assert "0.91" in finding.explanation

    # ── No-trigger ────────────────────────────────────────────────────────

    def test_does_not_trigger_on_clean_input(self) -> None:
        # scam_corpus_similarity defaults to 0.0 - no feedback yet means no effect
        finding = self.rule.evaluate(_clean())
        assert finding.triggered is False

    def test_does_not_trigger_below_threshold(self) -> None:
        finding = self.rule.evaluate(RuleInput(scam_corpus_similarity=0.50))
        assert finding.triggered is False

    def test_non_triggered_weight_still_set(self) -> None:
        finding = self.rule.evaluate(_clean())
        assert finding.weight == pytest.approx(0.70)

    # ── Weight / threshold configurability ──────────────────────────────────

    def test_custom_weight_reflected_in_finding(self) -> None:
        rule = SimilarToConfirmedScamRule(config=_cfg(similar_to_confirmed_scam=0.40))
        finding = rule.evaluate(RuleInput(scam_corpus_similarity=0.95))
        assert finding.weight == pytest.approx(0.40)
        assert finding.triggered is True

    def test_custom_threshold_reflected(self) -> None:
        rule = SimilarToConfirmedScamRule(
            config=_cfg_thresholds(scam_corpus_similarity_threshold=0.60)
        )
        finding = rule.evaluate(RuleInput(scam_corpus_similarity=0.65))
        assert finding.triggered is True

    # ── Return type ───────────────────────────────────────────────────────

    def test_returns_rule_finding_instance(self) -> None:
        assert isinstance(self.rule.evaluate(_clean()), RuleFinding)


# ===========================================================================
# TestNgoFundraisingStipendNetworkRule
# ===========================================================================

class TestNgoFundraisingStipendNetworkRule:

    def setup_method(self) -> None:
        self.rule = NgoFundraisingStipendNetworkRule()

    # ── Trigger ───────────────────────────────────────────────────────────

    def test_triggers_when_ngo_stipend_network_true(self) -> None:
        inp = RuleInput(ngo_stipend_network=True, ngo_stipend_network_company_count=5)
        finding = self.rule.evaluate(inp)
        assert finding.triggered is True

    def test_triggered_rule_id_correct(self) -> None:
        finding = self.rule.evaluate(RuleInput(ngo_stipend_network=True))
        assert finding.rule_id == "ngo_fundraising_stipend_network"

    def test_triggered_weight_matches_config_default(self) -> None:
        finding = self.rule.evaluate(RuleInput(ngo_stipend_network=True))
        assert finding.weight == pytest.approx(0.65)

    def test_triggered_explanation_mentions_company_count(self) -> None:
        finding = self.rule.evaluate(
            RuleInput(ngo_stipend_network=True, ngo_stipend_network_company_count=7)
        )
        assert "7" in finding.explanation

    # ── No-trigger ────────────────────────────────────────────────────────

    def test_does_not_trigger_on_clean_input(self) -> None:
        finding = self.rule.evaluate(_clean())
        assert finding.triggered is False

    def test_does_not_trigger_when_false(self) -> None:
        finding = self.rule.evaluate(RuleInput(ngo_stipend_network=False))
        assert finding.triggered is False

    def test_non_triggered_weight_still_set(self) -> None:
        finding = self.rule.evaluate(_clean())
        assert finding.weight == pytest.approx(0.65)

    def test_non_triggered_explanation_non_empty(self) -> None:
        assert self.rule.evaluate(_clean()).explanation != ""

    # ── Weight configurability ────────────────────────────────────────────

    def test_custom_weight_reflected_in_finding(self) -> None:
        rule = NgoFundraisingStipendNetworkRule(config=_cfg(ngo_fundraising_stipend_network=0.50))
        finding = rule.evaluate(RuleInput(ngo_stipend_network=True))
        assert finding.weight == pytest.approx(0.50)
        assert finding.triggered is True

    def test_weight_change_does_not_affect_trigger_logic(self) -> None:
        rule_low = NgoFundraisingStipendNetworkRule(config=_cfg(ngo_fundraising_stipend_network=0.10))
        rule_high = NgoFundraisingStipendNetworkRule(config=_cfg(ngo_fundraising_stipend_network=0.99))
        for inp in (RuleInput(ngo_stipend_network=True), RuleInput(ngo_stipend_network=False)):
            assert rule_low.evaluate(inp).triggered == rule_high.evaluate(inp).triggered

    # ── Return type ───────────────────────────────────────────────────────

    def test_returns_rule_finding_instance(self) -> None:
        assert isinstance(self.rule.evaluate(_clean()), RuleFinding)


# ===========================================================================
# TestStipendPerkContradictionRule
# ===========================================================================

class TestStipendPerkContradictionRule:

    def setup_method(self) -> None:
        self.rule = StipendPerkContradictionRule()

    # ── Trigger ───────────────────────────────────────────────────────────

    def test_triggers_when_perk_consistency_false(self) -> None:
        # perk_consistency_ok=False means contradiction found
        inp = RuleInput(perk_consistency_ok=False)
        assert self.rule.evaluate(inp).triggered is True

    def test_triggered_rule_id_correct(self) -> None:
        finding = self.rule.evaluate(RuleInput(perk_consistency_ok=False))
        assert finding.rule_id == "stipend_perk_contradiction"

    def test_triggered_weight_matches_config_default(self) -> None:
        finding = self.rule.evaluate(RuleInput(perk_consistency_ok=False))
        assert finding.weight == pytest.approx(0.35)

    def test_triggered_explanation_mentions_unpaid_or_stipend(self) -> None:
        explanation_lc = self.rule.evaluate(
            RuleInput(perk_consistency_ok=False)
        ).explanation.lower()
        assert any(kw in explanation_lc for kw in ("unpaid", "stipend", "contradiction"))

    # ── No-trigger ────────────────────────────────────────────────────────

    def test_does_not_trigger_when_consistent(self) -> None:
        # Default: perk_consistency_ok=True → no issue
        assert self.rule.evaluate(_clean()).triggered is False

    def test_does_not_trigger_when_perk_consistency_ok_true(self) -> None:
        assert self.rule.evaluate(RuleInput(perk_consistency_ok=True)).triggered is False

    def test_non_triggered_explanation_non_empty(self) -> None:
        assert self.rule.evaluate(_clean()).explanation != ""

    # ── Weight configurability ────────────────────────────────────────────

    def test_custom_weight_reflected_in_finding(self) -> None:
        rule = StipendPerkContradictionRule(config=_cfg(stipend_perk_contradiction=0.60))
        finding = rule.evaluate(RuleInput(perk_consistency_ok=False))
        assert finding.weight == pytest.approx(0.60)
        assert finding.triggered is True

    def test_weight_zero_still_triggers(self) -> None:
        rule = StipendPerkContradictionRule(config=_cfg(stipend_perk_contradiction=0.0))
        finding = rule.evaluate(RuleInput(perk_consistency_ok=False))
        assert finding.triggered is True
        assert finding.weight == pytest.approx(0.0)

    # ── Return type ───────────────────────────────────────────────────────

    def test_returns_rule_finding_instance(self) -> None:
        assert isinstance(self.rule.evaluate(_clean()), RuleFinding)



# ===========================================================================
# TestCrossCompanyDuplicateRule
# ===========================================================================

class TestCrossCompanyDuplicateRule:

    def setup_method(self) -> None:
        self.rule = CrossCompanyDuplicateRule()

    # ── Trigger ───────────────────────────────────────────────────────────

    def test_triggers_when_cross_company_duplicate_true(self) -> None:
        inp = RuleInput(cross_company_duplicate=True)
        assert self.rule.evaluate(inp).triggered is True

    def test_triggered_rule_id_correct(self) -> None:
        finding = self.rule.evaluate(RuleInput(cross_company_duplicate=True))
        assert finding.rule_id == "cross_company_duplicate"

    def test_triggered_weight_matches_config_default(self) -> None:
        finding = self.rule.evaluate(RuleInput(cross_company_duplicate=True))
        assert finding.weight == pytest.approx(0.80)

    def test_triggered_explanation_mentions_shell_or_duplicate(self) -> None:
        explanation_lc = self.rule.evaluate(
            RuleInput(cross_company_duplicate=True)
        ).explanation.lower()
        assert any(kw in explanation_lc for kw in ("shell", "duplicate", "company"))

    def test_triggered_explanation_mentions_allowlist_caveat(self) -> None:
        # Phase 1: allowlist caveat must be present (NayePankh false-positive note)
        explanation = self.rule.evaluate(RuleInput(cross_company_duplicate=True)).explanation
        assert "allowlist" in explanation.lower()

    # ── No-trigger ────────────────────────────────────────────────────────

    def test_does_not_trigger_on_clean_input(self) -> None:
        assert self.rule.evaluate(_clean()).triggered is False

    def test_does_not_trigger_when_false(self) -> None:
        assert self.rule.evaluate(RuleInput(cross_company_duplicate=False)).triggered is False

    def test_non_triggered_weight_still_set(self) -> None:
        assert self.rule.evaluate(_clean()).weight == pytest.approx(0.80)

    # ── Weight configurability ────────────────────────────────────────────

    def test_custom_weight_reflected_in_finding(self) -> None:
        rule = CrossCompanyDuplicateRule(config=_cfg(cross_company_duplicate=0.55))
        finding = rule.evaluate(RuleInput(cross_company_duplicate=True))
        assert finding.weight == pytest.approx(0.55)
        assert finding.triggered is True

    def test_trigger_logic_independent_of_weight(self) -> None:
        rule_low  = CrossCompanyDuplicateRule(config=_cfg(cross_company_duplicate=0.01))
        rule_high = CrossCompanyDuplicateRule(config=_cfg(cross_company_duplicate=0.99))
        assert rule_low.evaluate(RuleInput(cross_company_duplicate=True)).triggered is True
        assert rule_high.evaluate(RuleInput(cross_company_duplicate=False)).triggered is False

    # ── Return type ───────────────────────────────────────────────────────

    def test_returns_rule_finding_instance(self) -> None:
        assert isinstance(self.rule.evaluate(_clean()), RuleFinding)



# ===========================================================================
# TestExtremeStipendOutlierRule
# ===========================================================================

class TestExtremeStipendOutlierRule:

    def setup_method(self) -> None:
        self.rule = ExtremeStipendOutlierRule()

    # ── Trigger: HIGH outlier ─────────────────────────────────────────────

    def test_triggers_on_high_zscore(self) -> None:
        inp = RuleInput(stipend_peer_zscore=4.0)  # |4.0| > 3.0 default threshold
        assert self.rule.evaluate(inp).triggered is True

    def test_triggered_high_explanation_mentions_high(self) -> None:
        finding = self.rule.evaluate(RuleInput(stipend_peer_zscore=4.0))
        assert "high" in finding.explanation.lower()

    def test_triggered_high_explanation_mentions_lure(self) -> None:
        finding = self.rule.evaluate(RuleInput(stipend_peer_zscore=5.0))
        explanation_lc = finding.explanation.lower()
        assert any(kw in explanation_lc for kw in ("attract", "lure", "unrealistic"))

    # ── Trigger: LOW outlier ──────────────────────────────────────────────

    def test_triggers_on_low_zscore(self) -> None:
        inp = RuleInput(stipend_peer_zscore=-3.5)  # |-3.5| > 3.0
        assert self.rule.evaluate(inp).triggered is True

    def test_triggered_low_explanation_mentions_low(self) -> None:
        finding = self.rule.evaluate(RuleInput(stipend_peer_zscore=-4.0))
        assert "low" in finding.explanation.lower()

    def test_triggered_low_explanation_mentions_exploitative(self) -> None:
        finding = self.rule.evaluate(RuleInput(stipend_peer_zscore=-4.0))
        explanation_lc = finding.explanation.lower()
        assert any(kw in explanation_lc for kw in ("exploitat", "low-quality", "low"))

    def test_triggered_weight_matches_config_default(self) -> None:
        finding = self.rule.evaluate(RuleInput(stipend_peer_zscore=4.0))
        assert finding.weight == pytest.approx(0.45)

    def test_triggered_rule_id_correct(self) -> None:
        assert self.rule.evaluate(
            RuleInput(stipend_peer_zscore=4.0)
        ).rule_id == "extreme_stipend_outlier"

    # ── No-trigger ────────────────────────────────────────────────────────

    def test_does_not_trigger_within_threshold(self) -> None:
        # z = 2.9 is below default threshold of 3.0
        assert self.rule.evaluate(RuleInput(stipend_peer_zscore=2.9)).triggered is False

    def test_does_not_trigger_at_exact_threshold(self) -> None:
        # |z| == threshold → should NOT trigger (strictly greater than)
        assert self.rule.evaluate(RuleInput(stipend_peer_zscore=3.0)).triggered is False

    def test_does_not_trigger_negative_within_threshold(self) -> None:
        assert self.rule.evaluate(RuleInput(stipend_peer_zscore=-2.9)).triggered is False

    def test_does_not_trigger_on_zero_zscore(self) -> None:
        assert self.rule.evaluate(RuleInput(stipend_peer_zscore=0.0)).triggered is False

    def test_does_not_trigger_on_none_zscore(self) -> None:
        # None → unavailable peer data → no trigger
        assert self.rule.evaluate(RuleInput(stipend_peer_zscore=None)).triggered is False

    def test_none_zscore_explanation_mentions_unavailable(self) -> None:
        finding = self.rule.evaluate(RuleInput(stipend_peer_zscore=None))
        explanation_lc = finding.explanation.lower()
        assert any(kw in explanation_lc for kw in ("unavailable", "insufficient", "missing"))

    # ── Configurable threshold ────────────────────────────────────────────

    def test_custom_threshold_lower_triggers_sooner(self) -> None:
        rule = ExtremeStipendOutlierRule(
            config=_cfg_thresholds(stipend_zscore_threshold=2.0)
        )
        # z=2.5 is above 2.0 threshold → should trigger
        assert rule.evaluate(RuleInput(stipend_peer_zscore=2.5)).triggered is True

    def test_custom_threshold_higher_does_not_trigger_at_default(self) -> None:
        rule = ExtremeStipendOutlierRule(
            config=_cfg_thresholds(stipend_zscore_threshold=5.0)
        )
        # z=3.5 is below 5.0 threshold → should NOT trigger
        assert rule.evaluate(RuleInput(stipend_peer_zscore=3.5)).triggered is False

    def test_explanation_includes_actual_zscore(self) -> None:
        finding = self.rule.evaluate(RuleInput(stipend_peer_zscore=3.7))
        assert "3.7" in finding.explanation or "3.70" in finding.explanation

    def test_explanation_includes_threshold_value(self) -> None:
        finding = self.rule.evaluate(RuleInput(stipend_peer_zscore=4.0))
        assert "3.0" in finding.explanation or "±3" in finding.explanation

    # ── Weight configurability ────────────────────────────────────────────

    def test_custom_weight_reflected_in_finding(self) -> None:
        rule = ExtremeStipendOutlierRule(config=_cfg(extreme_stipend_outlier=0.30))
        finding = rule.evaluate(RuleInput(stipend_peer_zscore=4.0))
        assert finding.weight == pytest.approx(0.30)
        assert finding.triggered is True

    # ── Return type ───────────────────────────────────────────────────────

    def test_returns_rule_finding_instance(self) -> None:
        assert isinstance(self.rule.evaluate(_clean()), RuleFinding)

