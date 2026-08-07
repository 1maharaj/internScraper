"""
rules_engine.py
---------------
Fast, cheap, explainable first pass — deterministic rules evaluated before
any statistical or ML layers.

Design
------
Each rule is a callable that accepts a ``RuleInput`` dataclass and returns a
``RuleFinding`` (triggered, weight, explanation).  Rules are individually
instantiable and testable without running the full engine.

The ``RulesEngine`` runs a registered list of rules, collects all findings,
computes a combined score, and sets ``is_hard_reject`` when any finding
carries a weight above the hard-reject threshold.

Combined score formula
-----------------------
Noisy-OR combination: ``1 - ∏(1 - wᵢ)`` over all triggered rule weights.
This avoids the double-counting problem of simple summation while still
letting multiple moderate signals accumulate into a high score.

Weight rationale (all configurable in ``config.py`` without touching rules)
---------------------------------------------------------------------------
hard_disqualifying_signals  0.95  — near-automatic escalation
cross_company_duplicate      0.80  — same script / multiple shells
typosquat_domain             0.70  — off-platform domain mismatch
extreme_stipend_outlier      0.45  — directional anomaly (both directions)
mass_openings_vague_role     0.40  — combined heuristic
stipend_perk_contradiction   0.35  — data-quality / structural contradiction
unverifiable_company         0.10  — scraper noise; reduces CONFIDENCE only
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from scam_detector.config import Config, cfg as _default_cfg

# ---------------------------------------------------------------------------
# Input container — passed to every rule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuleInput:
    """
    All feature data a rule could need, gathered in one place.

    Fields mirror the feature vectors from Prompts 1–5.  Rules pull only
    what they need — unused fields are ignored.

    All fields default to safe/neutral values so rules can be tested with
    minimal synthetic inputs.
    """

    # ── Text features (Prompt 2) ──────────────────────────────────────────
    sensitive_info_requested: bool = False
    urgency_score: float = 0.0
    genericity_score: float = 0.0
    caps_ratio: float = 0.0
    exclamation_count: int = 0
    has_repeated_punctuation: bool = False
    summary_truncated: bool = False

    # ── Company features (Prompt 3) ───────────────────────────────────────
    company_is_suspect: bool = False          # category-leak flag from remediation
    typosquat_min_distance: float = 1.0       # 0 = exact brand match

    # ── URL features (Prompt 3) ───────────────────────────────────────────
    is_platform_internal: bool = False
    is_url_shortener: bool = False
    is_known_ats: bool = False
    domain_company_similarity: float = 1.0   # 1 = perfect match

    # ── Stipend features (Prompt 4) ───────────────────────────────────────
    stipend_peer_zscore: float | None = None
    perk_consistency_ok: bool = True          # True = consistent (no issue)

    # ── Structural features (Prompt 4) ────────────────────────────────────
    openings_zscore: float | None = None
    field_completeness: float = 1.0

    # ── Duplicate detection (Prompt 5) ────────────────────────────────────
    cross_company_duplicate: bool = False

    # ── Remediation flags (Prompt 1) ──────────────────────────────────────
    remediation_flags: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Rule protocol and finding
# ---------------------------------------------------------------------------


class RuleFinding(BaseModel):
    """A single triggered rule and its contribution."""

    rule_id: str = Field(description="Unique snake_case rule identifier")
    description: str = Field(description="Human-readable name")
    weight: float = Field(ge=0.0, le=1.0, description="Risk contribution if triggered")
    triggered: bool = Field(default=False)
    explanation: str = Field(default="", description="Why this rule fired (or didn't)")


@runtime_checkable
class Rule(Protocol):
    """
    Protocol every rule must satisfy.

    ``rule_id``  — unique snake_case identifier used in reports and config
    ``evaluate`` — takes RuleInput, returns RuleFinding
    """

    rule_id: str

    def evaluate(self, inp: RuleInput) -> RuleFinding:
        ...


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------


class RulesResult(BaseModel):
    """Aggregated output from the deterministic rules engine."""

    triggered: list[RuleFinding] = Field(default_factory=list)
    all_findings: list[RuleFinding] = Field(
        default_factory=list,
        description="All findings including non-triggered rules (for audit/debug)",
    )
    combined_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Noisy-OR combination of triggered rule weights",
    )
    is_hard_reject: bool = Field(
        default=False,
        description=(
            "True when any rule with weight ≥ hard_reject_threshold fired. "
            "Downstream scoring must treat this as near-certain escalation "
            "regardless of other signals."
        ),
    )
    triggered_rule_ids: list[str] = Field(default_factory=list)
    explanations: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Hard-reject threshold (above this weight → escalate to human review)
# ---------------------------------------------------------------------------

_HARD_REJECT_WEIGHT_THRESHOLD: float = 0.75


# ---------------------------------------------------------------------------
# Rule implementations
# ---------------------------------------------------------------------------


class HardDisqualifyingSignalsRule:
    """
    Rule 1: sensitive_info_request_detector == True

    Fires when the posting body asks for upfront payment, security deposit,
    Aadhaar/PAN number, bank account details, or similar.

    Weight: 0.95 (default) — near-automatic escalation to human review.
    This is treated as a near-hard disqualifying signal regardless of all
    other features.  Only a legitimate business reason confirmed by a human
    reviewer should override it.
    """

    rule_id = "hard_disqualifying_signals"

    def __init__(self, config: Config | None = None) -> None:
        self._cfg = config or _default_cfg

    def evaluate(self, inp: RuleInput) -> RuleFinding:
        w = self._cfg.rule_weights.hard_disqualifying_signals
        if inp.sensitive_info_requested:
            return RuleFinding(
                rule_id=self.rule_id,
                description="Hard disqualifying signals detected",
                weight=w,
                triggered=True,
                explanation=(
                    "Posting body contains requests for upfront payment, "
                    "security deposit, government ID (Aadhaar/PAN), or bank "
                    "account details — near-automatic escalation to human review."
                ),
            )
        return RuleFinding(
            rule_id=self.rule_id,
            description="Hard disqualifying signals detected",
            weight=w,
            triggered=False,
            explanation="No sensitive information requests detected in posting body.",
        )


class StipendPerkContradictionRule:
    """
    Rule 2: stipend_perk_consistency_check == True (inconsistency found)

    Fires when ``perk_consistency_ok`` is False, i.e. the stipend claims
    "unpaid" but perks include a compensation label like "Stipend".

    Weight: 0.35 — moderate signal; most likely a scraper template error
    rather than intentional fraud, but still degrades listing quality.
    """

    rule_id = "stipend_perk_contradiction"

    def __init__(self, config: Config | None = None) -> None:
        self._cfg = config or _default_cfg

    def evaluate(self, inp: RuleInput) -> RuleFinding:
        w = self._cfg.rule_weights.stipend_perk_contradiction
        # perk_consistency_ok=True means NO problem; False means contradiction
        if not inp.perk_consistency_ok:
            return RuleFinding(
                rule_id=self.rule_id,
                description="Stipend/perk internal contradiction",
                weight=w,
                triggered=True,
                explanation=(
                    "Stipend type is 'unpaid' but perks list includes a "
                    "monetary compensation label (e.g. 'Stipend'). "
                    "Likely a scraper template error; reduces listing reliability."
                ),
            )
        return RuleFinding(
            rule_id=self.rule_id,
            description="Stipend/perk internal contradiction",
            weight=w,
            triggered=False,
            explanation="Stipend type and perks are internally consistent.",
        )


class CrossCompanyDuplicateRule:
    """
    Rule 3: cross_company_duplicate_flag == True

    Fires when the same (or near-identical) posting text appears under a
    different company name elsewhere in the corpus.

    Weight: 0.80 — the strongest single fraud indicator.  The same scam
    script posted by multiple shell company names is very hard to fake
    accidentally.

    Note: see duplicate_detection.py for the NayePankh / Basti Ki Pathshala
    false-positive warning.  A same_parent_organization_allowlist should be
    applied before treating this as a hard reject.
    """

    rule_id = "cross_company_duplicate"

    def __init__(self, config: Config | None = None) -> None:
        self._cfg = config or _default_cfg

    def evaluate(self, inp: RuleInput) -> RuleFinding:
        w = self._cfg.rule_weights.cross_company_duplicate
        if inp.cross_company_duplicate:
            return RuleFinding(
                rule_id=self.rule_id,
                description="Cross-company near-duplicate text detected",
                weight=w,
                triggered=True,
                explanation=(
                    "This posting's text closely matches another listing "
                    "posted under a different company name — a strong indicator "
                    "of coordinated shell-company fraudulent posting. "
                    "Check same_parent_organization_allowlist before auto-rejecting."
                ),
            )
        return RuleFinding(
            rule_id=self.rule_id,
            description="Cross-company near-duplicate text detected",
            weight=w,
            triggered=False,
            explanation="No cross-company near-duplicate detected.",
        )


class ExtremeStipendOutlierRule:
    """
    Rule 4: |stipend_peer_zscore| > configurable threshold (default 3.0)

    Fires in both directions:
    - Suspiciously HIGH: unrealistically attractive to lure applicants
    - Suspiciously LOW:  potentially exploitative for the role category

    Weight: 0.45 — moderate, because context matters.  A high stipend at a
    funded startup is normal; the same at a blank-company posting is not.
    The explanation always states the direction.
    """

    rule_id = "extreme_stipend_outlier"

    def __init__(self, config: Config | None = None) -> None:
        self._cfg = config or _default_cfg

    def evaluate(self, inp: RuleInput) -> RuleFinding:
        w = self._cfg.rule_weights.extreme_stipend_outlier
        threshold = self._cfg.rule_thresholds.stipend_zscore_threshold

        z = inp.stipend_peer_zscore
        if z is None:
            return RuleFinding(
                rule_id=self.rule_id,
                description="Extreme stipend outlier (vs peer group)",
                weight=w,
                triggered=False,
                explanation=(
                    "Stipend z-score unavailable (performance-based stipend, "
                    "missing amount, or insufficient peer group)."
                ),
            )

        if abs(z) > threshold:
            direction = "HIGH" if z > 0 else "LOW"
            return RuleFinding(
                rule_id=self.rule_id,
                description="Extreme stipend outlier (vs peer group)",
                weight=w,
                triggered=True,
                explanation=(
                    f"Stipend is suspiciously {direction} for this role category "
                    f"(z-score = {z:.2f}, threshold = ±{threshold:.1f}). "
                    + (
                        "Unrealistically high stipends are used to attract applicants."
                        if direction == "HIGH"
                        else "Exploitatively low stipends signal a low-quality listing."
                    )
                ),
            )

        return RuleFinding(
            rule_id=self.rule_id,
            description="Extreme stipend outlier (vs peer group)",
            weight=w,
            triggered=False,
            explanation=f"Stipend is within normal range for peer group (z = {z:.2f}).",
        )


class UnverifiableCompanyRule:
    """
    Rule 5: is_company_suspect == True

    Fires when the company field contains a category/skill string rather
    than a real organisation name (e.g. company = "Digital Marketing").

    Weight: 0.10 — VERY LOW, and explicitly labelled as a CONFIDENCE
    reducer, not a fraud indicator.  Phase 1 established this reflects
    Internshala scraper noise where the company field was not present and
    the category leaked in.  Most of these are legitimate (if low-quality)
    postings, not scams.

    ⚠  Do NOT increase this weight.  The low weight is intentional and
       documents Phase 1's finding that this is scraper noise.
    """

    rule_id = "unverifiable_company"

    def __init__(self, config: Config | None = None) -> None:
        self._cfg = config or _default_cfg

    def evaluate(self, inp: RuleInput) -> RuleFinding:
        w = self._cfg.rule_weights.unverifiable_company
        if inp.company_is_suspect:
            return RuleFinding(
                rule_id=self.rule_id,
                description="Unverifiable company name (scraper noise)",
                weight=w,
                triggered=True,
                explanation=(
                    "Company field appears to contain a category or skill string "
                    "rather than a real organisation name (e.g. 'Digital Marketing'). "
                    "REDUCES CONFIDENCE IN COMPANY-IDENTITY FEATURES — does NOT "
                    "indicate fraud directly. Phase 1: this is scraper noise, not "
                    "employer intent."
                ),
            )
        return RuleFinding(
            rule_id=self.rule_id,
            description="Unverifiable company name (scraper noise)",
            weight=w,
            triggered=False,
            explanation="Company name appears to be a real organisation identifier.",
        )


class TyposquatDomainRule:
    """
    Rule 6: domain_company_similarity below threshold on an off-platform link

    Fires when the apply link domain does NOT resemble the stated company
    name AND the link is not a platform-internal or ATS link (those have
    legitimate reasons for domain mismatch).

    Weight: 0.70 — high.  A company saying "Razorpay" but linking to a
    random unrelated domain is a strong impersonation signal.

    Threshold: domain_company_similarity < 0.35 (configurable).
    """

    rule_id = "typosquat_domain"

    def __init__(self, config: Config | None = None) -> None:
        self._cfg = config or _default_cfg

    def evaluate(self, inp: RuleInput) -> RuleFinding:
        w = self._cfg.rule_weights.typosquat_domain
        threshold = self._cfg.rule_thresholds.typosquat_similarity_threshold

        # Skip if platform-internal, known ATS, or suspect company (identity unreliable)
        if inp.company_is_suspect or inp.is_platform_internal or inp.is_known_ats:
            return RuleFinding(
                rule_id=self.rule_id,
                description="Domain vs company name mismatch (off-platform)",
                weight=w,
                triggered=False,
                explanation=(
                    "Apply link is platform-internal or a known ATS, or the company "
                    "name is suspect/scraper noise — domain mismatch is expected or "
                    "unverifiable and not suspicious."
                ),
            )

        sim = inp.domain_company_similarity
        if sim < threshold:
            return RuleFinding(
                rule_id=self.rule_id,
                description="Domain vs company name mismatch (off-platform)",
                weight=w,
                triggered=True,
                explanation=(
                    f"Off-platform apply link domain does not resemble the stated "
                    f"company name (similarity = {sim:.2f}, threshold = {threshold:.2f}). "
                    "Possible brand impersonation or unrelated domain."
                ),
            )

        return RuleFinding(
            rule_id=self.rule_id,
            description="Domain vs company name mismatch (off-platform)",
            weight=w,
            triggered=False,
            explanation=(
                f"Domain reasonably matches company name (similarity = {sim:.2f})."
            ),
        )


class MassOpeningsVagueRoleRule:
    """
    Rule 7: openings_zscore HIGH AND genericity_score HIGH (combined condition)

    Fires only when BOTH conditions hold simultaneously — a single high
    opening count is normal for a large employer; a vague role title alone
    is common.  The combination of both is the signal: a company posting
    hundreds of openings for "Digital Marketing Intern" or "HR Intern" with
    no specificity is a shell-company pattern.

    Weight: 0.40 — moderate; the combined condition reduces false positives
    significantly compared to either flag alone.

    Thresholds (configurable):
        openings_zscore      > 2.0  (above 2 SD from peer mean)
        genericity_score     > 0.65 (highly generic title)
    """

    rule_id = "mass_openings_vague_role"

    def __init__(self, config: Config | None = None) -> None:
        self._cfg = config or _default_cfg

    def evaluate(self, inp: RuleInput) -> RuleFinding:
        w = self._cfg.rule_weights.mass_openings_vague_role
        oz_thresh = self._cfg.rule_thresholds.mass_openings_zscore_threshold
        gen_thresh = self._cfg.rule_thresholds.mass_openings_genericity_threshold

        oz = inp.openings_zscore
        gen = inp.genericity_score

        # If openings_zscore is unavailable, cannot evaluate the combined condition
        if oz is None:
            return RuleFinding(
                rule_id=self.rule_id,
                description="Mass openings with vague role (combined)",
                weight=w,
                triggered=False,
                explanation=(
                    "Openings z-score unavailable — combined condition cannot be evaluated."
                ),
            )

        openings_high = oz > oz_thresh
        role_vague = gen > gen_thresh

        if openings_high and role_vague:
            return RuleFinding(
                rule_id=self.rule_id,
                description="Mass openings with vague role (combined)",
                weight=w,
                triggered=True,
                explanation=(
                    f"Abnormally high opening count (z = {oz:.2f} > {oz_thresh:.1f}) "
                    f"combined with a vague/generic role title "
                    f"(genericity = {gen:.2f} > {gen_thresh:.2f}). "
                    "Shell companies often post mass vacancies for generic roles."
                ),
            )

        parts = []
        if not openings_high:
            parts.append(f"openings z-score {oz:.2f} ≤ {oz_thresh:.1f}")
        if not role_vague:
            parts.append(f"genericity {gen:.2f} ≤ {gen_thresh:.2f}")

        return RuleFinding(
            rule_id=self.rule_id,
            description="Mass openings with vague role (combined)",
            weight=w,
            triggered=False,
            explanation=(
                "Combined condition not met: " + "; ".join(parts) + "."
            ),
        )


# ---------------------------------------------------------------------------
# Default rule registry
# ---------------------------------------------------------------------------

def _default_rules(config: Config | None = None) -> list[Rule]:
    cfg = config or _default_cfg
    return [
        HardDisqualifyingSignalsRule(cfg),
        StipendPerkContradictionRule(cfg),
        CrossCompanyDuplicateRule(cfg),
        ExtremeStipendOutlierRule(cfg),
        UnverifiableCompanyRule(cfg),
        TyposquatDomainRule(cfg),
        MassOpeningsVagueRoleRule(cfg),
    ]


# ---------------------------------------------------------------------------
# RulesEngine
# ---------------------------------------------------------------------------


class RulesEngine:
    """
    Runs a registered list of rules against a ``RuleInput`` and returns a
    ``RulesResult``.

    Parameters
    ----------
    rules:
        Ordered list of rule instances.  Defaults to the full registry.
    config:
        Config override for weights / thresholds.  Defaults to module-level
        ``cfg`` singleton.
    hard_reject_threshold:
        Any rule whose weight is >= this value will set ``is_hard_reject``
        on the result when triggered.  Default: 0.75.
    """

    def __init__(
        self,
        rules: list[Rule] | None = None,
        config: Config | None = None,
        hard_reject_threshold: float = _HARD_REJECT_WEIGHT_THRESHOLD,
    ) -> None:
        effective_cfg = config or _default_cfg
        self._rules: list[Rule] = rules if rules is not None else _default_rules(effective_cfg)
        self._hard_reject_threshold = hard_reject_threshold

    def run(self, inp: RuleInput) -> RulesResult:
        """
        Evaluate all registered rules and aggregate findings.

        Score formula: Noisy-OR  ``1 - ∏(1 - wᵢ)`` over triggered weights.
        This prevents simple sum > 1 while letting multiple moderate signals
        compound realistically.
        """
        all_findings: list[RuleFinding] = []
        triggered: list[RuleFinding] = []
        is_hard_reject = False

        for rule in self._rules:
            finding = rule.evaluate(inp)
            all_findings.append(finding)
            if finding.triggered:
                triggered.append(finding)
                if finding.weight >= self._hard_reject_threshold:
                    is_hard_reject = True

        # Noisy-OR combination of triggered weights
        if triggered:
            product = 1.0
            for f in triggered:
                product *= 1.0 - f.weight
            combined_score = round(1.0 - product, 4)
        else:
            combined_score = 0.0

        return RulesResult(
            triggered=triggered,
            all_findings=all_findings,
            combined_score=combined_score,
            is_hard_reject=is_hard_reject,
            triggered_rule_ids=[f.rule_id for f in triggered],
            explanations=[f.explanation for f in triggered],
        )


# ---------------------------------------------------------------------------
# Legacy shim — keeps scoring/__init__.py and tests/test_scoring.py working
# ---------------------------------------------------------------------------

def apply_rules(features: object) -> RulesResult:
    """
    Evaluate all deterministic rules against *features*.

    Accepts either a ``RuleInput`` directly, or any object with feature
    attributes (duck-typed).  Falls back to a default ``RuleInput()`` for
    skeleton/stub usage so existing callers don't break.
    """
    if isinstance(features, RuleInput):
        inp = features
    else:
        # Duck-type bridge: pull fields from a FeatureVector-like object
        # so the legacy ``apply_rules(feature_vector)`` call pattern still works.
        def _get(obj: object, *attrs: str, default: object = None) -> object:
            for attr in attrs:
                try:
                    val = getattr(obj, attr)
                    if val is not None:
                        return val
                except AttributeError:
                    pass
            return default

        inp = RuleInput(
            sensitive_info_requested=bool(
                _get(features, "text.sensitive_info_requested",
                     "sensitive_info_requested", default=False)
            ),
            perk_consistency_ok=bool(
                _get(features, "stipend.perk_consistency_ok",
                     "perk_consistency_ok", default=True)
            ),
            cross_company_duplicate=bool(
                _get(features, "cross_company_duplicate", default=False)
            ),
            stipend_peer_zscore=None,
            company_is_suspect=bool(
                _get(features, "company.is_suspect", "is_suspect", default=False)
            ),
            domain_company_similarity=float(
                _get(features, "url.domain_company_similarity",
                     "domain_company_similarity", default=1.0)
            ),
            is_platform_internal=bool(
                _get(features, "url.is_platform_internal",
                     "is_platform_internal", default=False)
            ),
            is_known_ats=bool(
                _get(features, "url.is_known_ats", "is_known_ats", default=False)
            ),
            openings_zscore=None,
            genericity_score=float(
                _get(features, "text.genericity_score",
                     "genericity_score", default=0.0)
            ),
        )

    engine = RulesEngine()
    return engine.run(inp)
