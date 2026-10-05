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

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

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
    guaranteed_outcome_claim: bool = False
    external_form_detected: bool = False
    personal_contact_handoff_detected: bool = False
    training_program_disguised_as_internship: bool = False
    zero_shot_scam_category: str | None = None
    zero_shot_scam_confidence: float = 0.0
    scam_corpus_similarity: float = 0.0
    urgency_score: float = 0.0
    genericity_score: float = 0.0
    caps_ratio: float = 0.0
    exclamation_count: int = 0
    has_repeated_punctuation: bool = False
    summary_truncated: bool = False

    # ── Company features (Prompt 3) ───────────────────────────────────────
    company_is_suspect: bool = False          # category-leak flag from remediation
    typosquat_min_distance: float = 1.0       # 0 = exact brand match
    domain_age_days: int | None = None        # None = unknown/not applicable (platform/ATS link)

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

    # ── Graph features (Phase 2/3) ────────────────────────────────────────
    shared_infrastructure: bool = False

    # ── NGO / fundraising-sector coordinated stipend network ──────────────
    ngo_stipend_network: bool = False
    ngo_stipend_network_company_count: int = 0

    # ── Upfront payment & pay-to-work signals ─────────────────────────────
    payment_required: bool = False
    registration_fee: float = 0.0
    fake_certificate_offer: bool = False

    # ── Recruiter contact authenticity ────────────────────────────────────
    recruiter_email_type: str = "Corporate"
    suspicious_email_domain: bool = False

    # ── Psychological pressure & manipulation signals ─────────────────────
    emotional_manipulation_score: float = 0.0
    phishing_language_score: float = 0.0

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


class GuaranteedOutcomeClaimRule:
    """
    Rule 1b: guaranteed_outcome_claim_detector == True

    Fires when the posting claims a guaranteed job/placement/certificate
    outcome, skips the interview entirely, or funnels applicants to a
    WhatsApp/Telegram handoff — marketing hooks scammers front-load into the
    title/summary/perks, which survive scraper truncation better than
    payment-request language buried later in the text.

    Weight: 0.85 (default) — near-hard disqualifying. Legitimate employers
    cannot legally guarantee a hiring outcome or skip candidate screening.
    """

    rule_id = "guaranteed_outcome_claim"

    def __init__(self, config: Config | None = None) -> None:
        self._cfg = config or _default_cfg

    def evaluate(self, inp: RuleInput) -> RuleFinding:
        w = self._cfg.rule_weights.guaranteed_outcome_claim
        if inp.guaranteed_outcome_claim:
            return RuleFinding(
                rule_id=self.rule_id,
                description="Guaranteed-outcome / no-interview claim detected",
                weight=w,
                triggered=True,
                explanation=(
                    "Posting claims a guaranteed job/placement/certificate outcome, "
                    "no-interview instant selection, or a WhatsApp/Telegram-only "
                    "handoff — legitimate employers never guarantee hiring outcomes "
                    "or skip screening entirely."
                ),
            )
        return RuleFinding(
            rule_id=self.rule_id,
            description="Guaranteed-outcome / no-interview claim detected",
            weight=w,
            triggered=False,
            explanation="No guaranteed-outcome or no-interview claims detected.",
        )


class ExternalFormHandoffRule:
    """
    Rule 1c: posting routes applicants to a generic third-party form
    (Google Forms, Typeform, JotForm, etc.) and/or a personal phone number,
    instead of the platform's own application flow.

    Real-data basis: a "Zefrix" posting (company field corrupted to
    "Content") read verbatim: "Apply now: https://forms.gle/... or call on
    +918854996448" — informal WhatsApp-forward style. This is the classic
    funnel for "internship" postings that are actually lead-generation for a
    phone-sold paid training program: the applicant fills a generic form,
    gets called, and the actual pitch (and money request) happens verbally,
    off-platform — invisible to any text analysis of the original posting.
    The form link + phone number is the only trace this leaves.

    Weight: 0.60 when only one of (form link, phone handoff) is present;
    escalates toward the hard-reject band when BOTH appear together, since
    that combination has no ordinary legitimate explanation on an
    aggregator platform that already provides its own apply flow.
    """

    rule_id = "external_form_handoff"

    def __init__(self, config: Config | None = None) -> None:
        self._cfg = config or _default_cfg

    def evaluate(self, inp: RuleInput) -> RuleFinding:
        base_w = self._cfg.rule_weights.external_form_handoff
        form = inp.external_form_detected
        phone = inp.personal_contact_handoff_detected

        if form and phone:
            # Both together: escalate weight (capped at 1.0) — this specific
            # combination is what the real Zefrix example showed.
            w = min(1.0, base_w + 0.25)
            return RuleFinding(
                rule_id=self.rule_id,
                description="Off-platform form + personal phone handoff detected",
                weight=w,
                triggered=True,
                explanation=(
                    "Posting routes applicants to a generic third-party form AND a "
                    "personal phone number instead of the platform's own apply flow — "
                    "a classic funnel for phone-sold 'training program' scams disguised "
                    "as internships, where the actual pitch happens verbally off-platform."
                ),
            )
        if form or phone:
            which = "a generic third-party form link" if form else "a personal phone-number handoff"
            return RuleFinding(
                rule_id=self.rule_id,
                description="Off-platform form + personal phone handoff detected",
                weight=base_w,
                triggered=True,
                explanation=(
                    f"Posting includes {which} instead of relying on the platform's own "
                    f"apply flow — worth reviewing, though not conclusive on its own."
                ),
            )
        return RuleFinding(
            rule_id=self.rule_id,
            description="Off-platform form + personal phone handoff detected",
            weight=base_w,
            triggered=False,
            explanation="No external form link or personal phone handoff detected.",
        )


class TrainingProgramDisguisedAsInternshipRule:
    """
    Rule 1d: posting frames the internship itself as a packaged training or
    paid-admission product rather than real work.

    Real-data basis: a LetsIntern posting for "Electric Vehicle Design
    Internship" (company field identical to its own title — no real
    employer exists) describes itself as "a career-focused, hands-on
    training program", with an incoherent skill list (AWS/DevOps/WordPress
    for an EV design role) confirming templated course-marketing content.
    A second example ("BBA" as company) offers "100% sponsored admission to
    a professional degree or certification program" framed as an
    internship — recruiting students into paid degree/certification
    admissions.

    Weight: 0.65 — moderate, not a hard reject. Legitimate accelerator- or
    bootcamp-style internship programs that genuinely combine structured
    training with real work exist too; this surfaces the pattern for human
    review rather than presuming fraud outright.
    """

    rule_id = "training_program_disguised_as_internship"

    def __init__(self, config: Config | None = None) -> None:
        self._cfg = config or _default_cfg

    def evaluate(self, inp: RuleInput) -> RuleFinding:
        w = self._cfg.rule_weights.training_program_disguised_as_internship
        if inp.training_program_disguised_as_internship:
            return RuleFinding(
                rule_id=self.rule_id,
                description="Internship framed as a training/certification product",
                weight=w,
                triggered=True,
                explanation=(
                    "Posting frames the internship itself as a packaged training "
                    "program or paid degree/certification admission rather than "
                    "real work — a common funnel for phone-sold training fees "
                    "disguised as an internship opportunity."
                ),
            )
        return RuleFinding(
            rule_id=self.rule_id,
            description="Internship framed as a training/certification product",
            weight=w,
            triggered=False,
            explanation="No training/certification-product framing detected.",
        )


class ZeroShotSemanticScamRule:
    """
    Rule 1e: pretrained zero-shot NLI classifier flags the posting as
    matching a known scam-behavior category, above a confidence threshold.

    ⚠️  Off by default and NOT currently recommended for use. A small
    hand-picked validation looked promising, but a follow-up test against a
    random 250-record real sample found a ~20% false-positive rate,
    dominated by one label ("sells a paid training or certification course
    disguised as a job") over-triggering on completely ordinary postings
    from real companies. See ``ZeroShotConfig`` in config.py for the full
    finding before ever setting ``cfg.zero_shot.enabled = True``.

    Unlike every other text rule in this engine, this signal is NOT a
    hand-written regex pattern — it's a pretrained model's semantic
    judgment. When disabled, ``inp.zero_shot_scam_category`` is always None
    and this rule never triggers (i.e. it is a safe no-op today).

    Weight: 0.55 (default) — deliberately lower than the regex-based rules
    in this tier, since a probabilistic model judgment is inherently less
    certain than an exact pattern match on verified real examples.
    """

    rule_id = "zero_shot_semantic_scam_signal"

    def __init__(self, config: Config | None = None) -> None:
        self._cfg = config or _default_cfg

    def evaluate(self, inp: RuleInput) -> RuleFinding:
        w = self._cfg.rule_weights.zero_shot_semantic_scam_signal
        threshold = self._cfg.zero_shot.confidence_threshold
        if inp.zero_shot_scam_category and inp.zero_shot_scam_confidence >= threshold:
            return RuleFinding(
                rule_id=self.rule_id,
                description="Pretrained model flagged a semantic scam-behavior pattern",
                weight=w,
                triggered=True,
                explanation=(
                    f"Zero-shot semantic classifier matched this posting to "
                    f"'{inp.zero_shot_scam_category}' with "
                    f"{inp.zero_shot_scam_confidence:.0%} confidence — a pretrained-model "
                    f"signal, not a hand-written pattern match, useful for catching "
                    f"scam phrasing that regex-based rules would miss."
                ),
            )
        return RuleFinding(
            rule_id=self.rule_id,
            description="Pretrained model flagged a semantic scam-behavior pattern",
            weight=w,
            triggered=False,
            explanation="No semantic scam-behavior pattern detected (or classifier disabled).",
        )


class SimilarToConfirmedScamRule:
    """
    Rule 1f: posting is semantically near-identical to a listing a human
    moderator already confirmed as a scam (the feedback loop).

    This is the mechanism that makes confirmed scams generalize beyond the
    same company reposting under its own name (which ``company_reputation_
    score`` already catches) to a DIFFERENT company using similar wording -
    e.g. a template scam reused by a new shell company. SBERT cosine
    similarity is computed against every confirmed_scam record in
    FeedbackStore (see ``get_scam_corpus_embeddings`` /
    ``scam_corpus_similarity`` in text_features.py) - this was previously
    computed every run but silently discarded, feeding neither this rule
    nor the anomaly model.

    Weight: 0.70 (default) — high, since it's grounded in verified human
    ground truth rather than a heuristic pattern, but deliberately kept
    below the 0.75 hard-reject threshold: semantic similarity at this
    threshold is strong but not infallible (two unrelated postings in a
    narrow category, e.g. both generic "data entry" roles, could
    legitimately score high without being the same scam).

    Threshold calibration (0.85, measured with all-MiniLM-L6-v2): tested
    whether a lower threshold would catch more reworded clones without
    false-flagging unrelated postings — it would not. A near-identical scam
    clone with one word changed scored 0.998; a heavily-reworded paraphrase
    of the same underlying template scored only 0.61 (would evade almost
    any reasonable threshold — an inherent limit of semantic similarity for
    determined rewording, not a bug). Critically, an UNRELATED legitimate
    posting (different company, same job category as a confirmed scam)
    scored 0.71 similarity to that scam — higher than two unrelated
    legitimate postings scored against each other (0.50). A threshold
    anywhere near 0.71 would false-flag unrelated legitimate postings for
    being topically similar. 0.85 sits safely above that false-positive
    zone while still catching near-identical/lightly-reworded clones.

    Empty/no feedback yet: ``scam_corpus_similarity`` defaults to 0.0 and
    this rule never triggers until at least one confirmed_scam label
    exists - it has no effect before the feedback loop has any data.
    """

    rule_id = "similar_to_confirmed_scam"

    def __init__(self, config: Config | None = None) -> None:
        self._cfg = config or _default_cfg

    def evaluate(self, inp: RuleInput) -> RuleFinding:
        w = self._cfg.rule_weights.similar_to_confirmed_scam
        threshold = self._cfg.rule_thresholds.scam_corpus_similarity_threshold
        sim = inp.scam_corpus_similarity

        if sim >= threshold:
            return RuleFinding(
                rule_id=self.rule_id,
                description="Near-identical to a human-confirmed scam posting",
                weight=w,
                triggered=True,
                explanation=(
                    f"This posting is semantically near-identical (similarity = "
                    f"{sim:.2f}, threshold = {threshold:.2f}) to a listing a "
                    f"moderator already confirmed was a scam — likely the same "
                    f"template reused under a different company name."
                ),
            )

        return RuleFinding(
            rule_id=self.rule_id,
            description="Near-identical to a human-confirmed scam posting",
            weight=w,
            triggered=False,
            explanation=(
                f"Not closely similar to any confirmed scam on record "
                f"(similarity = {sim:.2f})."
            ),
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


class SharedInfrastructureRule:
    """
    Rule 3b: shared_infrastructure_flag == True

    Fires when a company's applyLink domain is shared with 3+ OTHER distinctly-named
    companies in the corpus. This indicates shared, coordinate posting infrastructure,
    which is a strong signal for automated shell-posting networks.

    Weight: 0.65 (default, configurable)
    """

    rule_id = "shared_infrastructure"

    def __init__(self, config: Config | None = None) -> None:
        self._cfg = config or _default_cfg

    def evaluate(self, inp: RuleInput) -> RuleFinding:
        w = self._cfg.rule_weights.shared_infrastructure
        if inp.shared_infrastructure:
            return RuleFinding(
                rule_id=self.rule_id,
                description="Shared off-platform infrastructure across multiple companies detected",
                weight=w,
                triggered=True,
                explanation=(
                    "This company's applyLink domain is shared with 3+ other distinctly-named "
                    "companies — a strong signature of coordinated shell-company posting networks."
                ),
            )
        return RuleFinding(
            rule_id=self.rule_id,
            description="Shared off-platform infrastructure across multiple companies detected",
            weight=w,
            triggered=False,
            explanation="No shared off-platform infrastructure detected.",
        )


class NgoFundraisingStipendNetworkRule:
    """
    Rule 3c: NGO/fundraising-sector posting whose exact lump-sum stipend
    amount is shared by 3+ distinct companies in the corpus.

    Empirical basis: on real scraped data, several "Foundation"-named
    entities post under different role titles (Business Consultant, Program
    Assistant, Fundraising, Crowdfunding) but share an identical lump-sum
    stipend (e.g. ₹15,000) — a templated-payout signature that SBERT-based
    text-duplicate detection does not catch, since each posting's wording
    is independently written.

    Deliberately scoped to lump-sum stipends only (not monthly, which
    legitimately clusters around common round numbers across unrelated
    real companies) and to NGO/fundraising-sector postings only — a
    coincidental stipend match at an ordinary tech company is not this
    signal.

    Weight: 0.65 (default) — moderate, NOT a hard reject. Legitimate
    fundraising-platform aggregators that standardize pay across multiple
    NGO partners are a real business model too (same ambiguity already
    documented for cross_company_duplicate's NayePankh/Basti Ki Pathshala
    case) — this surfaces the pattern for human review, it does not
    presume fraud.
    """

    rule_id = "ngo_fundraising_stipend_network"

    def __init__(self, config: Config | None = None) -> None:
        self._cfg = config or _default_cfg

    def evaluate(self, inp: RuleInput) -> RuleFinding:
        w = self._cfg.rule_weights.ngo_fundraising_stipend_network
        if inp.ngo_stipend_network:
            return RuleFinding(
                rule_id=self.rule_id,
                description="NGO/fundraising-sector coordinated stipend network detected",
                weight=w,
                triggered=True,
                explanation=(
                    f"This NGO/fundraising-sector posting shares its exact lump-sum "
                    f"stipend amount with {inp.ngo_stipend_network_company_count} other "
                    f"distinctly-named companies — a templated-payout pattern consistent "
                    f"with a coordinated shell-NGO network, though also seen with "
                    f"legitimate fundraising-platform aggregators. Recommend human review "
                    f"of the company's registration/legitimacy."
                ),
            )
        return RuleFinding(
            rule_id=self.rule_id,
            description="NGO/fundraising-sector coordinated stipend network detected",
            weight=w,
            triggered=False,
            explanation="No coordinated NGO stipend network pattern detected.",
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


class YoungDomainAgeRule:
    """
    Rule 6b: off-platform employer domain was registered very recently.

    A classic, well-established fraud signal: scam operations typically
    register a domain shortly before (or even after) starting to post
    fraudulent listings, while legitimate companies' domains are usually
    years old. ``inp.domain_age_days`` is only ever populated for a genuine
    off-platform employer domain — the WHOIS lookup is skipped entirely for
    platform/ATS links (see ``extract_company_url_features`` in
    company_features.py), since those are always old and identical across
    nearly every record on that platform, carrying zero discriminative
    value. ``None`` means unknown/not-applicable and never triggers.

    Weight: 0.55 (default) — moderate. A young domain alone is not
    definitive (genuine new startups have young domains too); this is a
    supporting signal, not a hard reject.

    NOTE: as of this writing, this rule is structurally unreachable on the
    current real corpus — every record's apply link routes through a
    platform or known ATS (confirmed empirically across 2,000+ real
    records), so ``domain_age_days`` is always None. It is correct, tested,
    and ready for the moment genuine off-platform employer links appear in
    the data (e.g. broader scraper coverage, or direct employer postings).
    """

    rule_id = "young_domain_age"

    def __init__(self, config: Config | None = None) -> None:
        self._cfg = config or _default_cfg

    def evaluate(self, inp: RuleInput) -> RuleFinding:
        w = self._cfg.rule_weights.young_domain_age
        threshold_days = self._cfg.rule_thresholds.young_domain_age_days_threshold

        if inp.domain_age_days is None:
            return RuleFinding(
                rule_id=self.rule_id,
                description="Off-platform employer domain registered very recently",
                weight=w,
                triggered=False,
                explanation="Domain age unknown or not applicable (platform/ATS link).",
            )

        if inp.domain_age_days < threshold_days:
            return RuleFinding(
                rule_id=self.rule_id,
                description="Off-platform employer domain registered very recently",
                weight=w,
                triggered=True,
                explanation=(
                    f"Off-platform employer domain was registered only "
                    f"{inp.domain_age_days} days ago (threshold = {threshold_days}) — "
                    "a classic fraud signal, though not definitive on its own."
                ),
            )

        return RuleFinding(
            rule_id=self.rule_id,
            description="Off-platform employer domain registered very recently",
            weight=w,
            triggered=False,
            explanation=f"Domain age ({inp.domain_age_days} days) is not suspiciously young.",
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


class UpfrontFeeAndPayToWorkRule:
    """
    Rule 9: Upfront payment, registration fee, or pay-to-work pattern.

    Fires when any of:
      - payment_required is True
      - registration_fee > 0
      - fake_certificate_offer is True (guaranteed certificate upon payment)

    Weight: 0.90 — near hard-reject; genuine internships never charge candidates
    for registration, application, security deposits, or certificates.
    """

    rule_id = "upfront_fee_and_pay_to_work"

    def __init__(self, config: Config | None = None) -> None:
        self._cfg = config or _default_cfg

    def evaluate(self, inp: RuleInput) -> RuleFinding:
        w = self._cfg.rule_weights.upfront_fee_and_pay_to_work
        reasons: list[str] = []
        if inp.payment_required:
            reasons.append("upfront payment explicitly required")
        if inp.registration_fee > 0:
            reasons.append(f"registration fee of INR {inp.registration_fee:.2f} demanded")
        if inp.fake_certificate_offer:
            reasons.append("pay-to-receive certificate / certificate sales pattern")

        if reasons:
            return RuleFinding(
                rule_id=self.rule_id,
                description="Upfront fee or pay-to-work requirement",
                weight=w,
                triggered=True,
                explanation=(
                    "Pay-to-work pattern detected: " + "; ".join(reasons) +
                    ". Legitimate employers do not charge internship candidates fees or deposits."
                ),
            )
        return RuleFinding(
            rule_id=self.rule_id,
            description="Upfront fee or pay-to-work requirement",
            weight=w,
            triggered=False,
            explanation="No upfront fees or pay-to-work demands detected.",
        )


class SuspiciousRecruiterContactRule:
    """
    Rule 10: Suspicious recruiter contact channel / disposable email.

    Fires when recruiter uses a free webmail service (e.g. Gmail, Yahoo)
    or an explicitly flagged suspicious email domain.

    Weight: 0.50 — moderate signal. Legitimate corporate recruiters use verified
    company domains; free email addresses are frequently used in scam operations.
    """

    rule_id = "suspicious_recruiter_contact"

    def __init__(self, config: Config | None = None) -> None:
        self._cfg = config or _default_cfg

    def evaluate(self, inp: RuleInput) -> RuleFinding:
        w = self._cfg.rule_weights.suspicious_recruiter_contact
        reasons: list[str] = []
        if inp.suspicious_email_domain:
            reasons.append("recruiter email domain flagged as suspicious/unverified")
        if str(inp.recruiter_email_type).strip().lower() in ("free", "disposable"):
            reasons.append(f"recruiter uses {inp.recruiter_email_type.lower()} webmail instead of corporate domain")

        if reasons:
            return RuleFinding(
                rule_id=self.rule_id,
                description="Suspicious recruiter contact channel",
                weight=w,
                triggered=True,
                explanation=(
                    "Recruiter contact risk: " + "; ".join(reasons) +
                    ". Corporate listings should be sourced from verified enterprise domains."
                ),
            )
        return RuleFinding(
            rule_id=self.rule_id,
            description="Suspicious recruiter contact channel",
            weight=w,
            triggered=False,
            explanation="Recruiter contact channel appears legitimate or corporate-affiliated.",
        )


class UrgencyAndPsychologicalPressureRule:
    """
    Rule 11: Artificial urgency & psychological pressure.

    Fires when high urgency score, emotional manipulation score, or phishing language
    score exceeds configured thresholds.

    Weight: 0.45 — moderate signal; scams frequently use artificial countdowns,
    pressure tactics ('apply in 2 hours', 'only 1 slot left') to rush applicants.
    """

    rule_id = "urgency_psychological_pressure"

    def __init__(self, config: Config | None = None) -> None:
        self._cfg = config or _default_cfg

    def evaluate(self, inp: RuleInput) -> RuleFinding:
        w = self._cfg.rule_weights.urgency_psychological_pressure
        urg_thresh = self._cfg.rule_thresholds.urgency_score_threshold
        emo_thresh = self._cfg.rule_thresholds.emotional_manipulation_threshold

        reasons: list[str] = []
        # Normalized urgency score (if scaled 0-100, normalize to [0, 1])
        urg = inp.urgency_score / 100.0 if inp.urgency_score > 1.0 else inp.urgency_score
        emo = inp.emotional_manipulation_score / 100.0 if inp.emotional_manipulation_score > 1.0 else inp.emotional_manipulation_score
        phish = inp.phishing_language_score / 100.0 if inp.phishing_language_score > 1.0 else inp.phishing_language_score

        if urg > urg_thresh:
            reasons.append(f"high artificial urgency score ({urg:.2f} > {urg_thresh:.2f})")
        if emo > emo_thresh:
            reasons.append(f"emotional manipulation score ({emo:.2f} > {emo_thresh:.2f})")
        if phish > 0.50:
            reasons.append(f"phishing language indicators ({phish:.2f} > 0.50)")

        if reasons:
            return RuleFinding(
                rule_id=self.rule_id,
                description="Artificial urgency and psychological manipulation",
                weight=w,
                triggered=True,
                explanation=(
                    "Psychological pressure tactics detected: " + "; ".join(reasons) +
                    ". Scammers use fabricated urgency to prevent thorough vetting."
                ),
            )
        return RuleFinding(
            rule_id=self.rule_id,
            description="Artificial urgency and psychological manipulation",
            weight=w,
            triggered=False,
            explanation="Urgency and emotional indicators within normal bounds.",
        )


# ---------------------------------------------------------------------------
# Default rule registry
# ---------------------------------------------------------------------------

def _default_rules(config: Config | None = None) -> list[Rule]:
    cfg = config or _default_cfg
    return [
        HardDisqualifyingSignalsRule(cfg),
        GuaranteedOutcomeClaimRule(cfg),
        ExternalFormHandoffRule(cfg),
        TrainingProgramDisguisedAsInternshipRule(cfg),
        ZeroShotSemanticScamRule(cfg),
        SimilarToConfirmedScamRule(cfg),
        UpfrontFeeAndPayToWorkRule(cfg),
        StipendPerkContradictionRule(cfg),
        CrossCompanyDuplicateRule(cfg),
        SharedInfrastructureRule(cfg),
        NgoFundraisingStipendNetworkRule(cfg),
        SuspiciousRecruiterContactRule(cfg),
        UrgencyAndPsychologicalPressureRule(cfg),
        ExtremeStipendOutlierRule(cfg),
        UnverifiableCompanyRule(cfg),
        TyposquatDomainRule(cfg),
        YoungDomainAgeRule(cfg),
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
        def _get(obj: object, *attrs: str, default: Any = None) -> Any:
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
            shared_infrastructure=bool(
                _get(features, "shared_infrastructure", default=False)
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
