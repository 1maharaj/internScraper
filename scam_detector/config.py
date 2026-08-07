"""
config.py
---------
Single source of truth for all tunable knobs:
thresholds, weights, and feature flags.

Import anywhere via:
    from scam_detector.config import cfg
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ScoringWeights(BaseModel):
    """Relative weights applied when combining individual risk signals."""

    text_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    company_weight: float = Field(default=0.20, ge=0.0, le=1.0)
    url_weight: float = Field(default=0.20, ge=0.0, le=1.0)
    stipend_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    temporal_weight: float = Field(default=0.10, ge=0.0, le=1.0)
    structural_weight: float = Field(default=0.10, ge=0.0, le=1.0)


class BlendWeights(BaseModel):
    """
    Weights for blending rules-engine score with anomaly-model score.

    Default: 60% rules / 40% anomaly.

    Rationale: rules are more precise early on because they are grounded in
    Phase 1's actual corpus findings.  The anomaly-model weight should
    increase over time as more data accumulates and the IsolationForest
    (or a future supervised model) becomes better calibrated — tune
    ``anomaly_weight`` upward (and ``rules_weight`` downward) without
    touching scoring code.
    """

    rules_weight: float = Field(default=0.60, ge=0.0, le=1.0)
    anomaly_weight: float = Field(default=0.40, ge=0.0, le=1.0)


class RuleWeights(BaseModel):
    """
    Per-rule weights for the deterministic rules engine.

    Each value is the risk contribution [0, 1] added when the rule fires.
    Tune here without touching rule logic.

    Rationale behind defaults
    -------------------------
    hard_disqualifying_signals : 0.95  — near-automatic escalation to human
                                         review; only 1 other signal can override
    cross_company_duplicate     : 0.80  — same script / multiple shells is the
                                         strongest single fraud indicator
    typosquat_domain            : 0.70  — off-platform mismatch is strong
    extreme_stipend_outlier     : 0.45  — both directions are suspicious but
                                         context matters; moderate weight
    mass_openings_vague_role    : 0.40  — combined heuristic; moderate
    stipend_perk_contradiction  : 0.35  — data quality issue, not certain fraud
    unverifiable_company        : 0.10  — scraper noise, very low weight;
                                         reduces CONFIDENCE, not fraud score
    """

    hard_disqualifying_signals: float = Field(default=0.95, ge=0.0, le=1.0)
    cross_company_duplicate: float = Field(default=0.80, ge=0.0, le=1.0)
    typosquat_domain: float = Field(default=0.70, ge=0.0, le=1.0)
    extreme_stipend_outlier: float = Field(default=0.45, ge=0.0, le=1.0)
    mass_openings_vague_role: float = Field(default=0.40, ge=0.0, le=1.0)
    stipend_perk_contradiction: float = Field(default=0.35, ge=0.0, le=1.0)
    unverifiable_company: float = Field(default=0.10, ge=0.0, le=1.0)


class RuleThresholds(BaseModel):
    """
    Per-rule numeric thresholds, configurable without touching rule logic.
    """

    # ExtremeStipendOutlierRule: |z-score| must exceed this to trigger
    stipend_zscore_threshold: float = Field(default=3.0, ge=0.0)

    # TyposquatDomainRule: domain_company_similarity below this on off-platform link
    typosquat_similarity_threshold: float = Field(default=0.35, ge=0.0, le=1.0)

    # MassOpeningsVagueRoleRule: openings z-score above this AND genericity above this
    mass_openings_zscore_threshold: float = Field(default=2.0, ge=0.0)
    mass_openings_genericity_threshold: float = Field(default=0.65, ge=0.0, le=1.0)


class Thresholds(BaseModel):
    """
    Legacy 0–1 decision boundaries (kept for older callers).

    Prefer :class:`DecisionThresholds` for the Phase 5 scam_score (0–100) path.
    """

    auto_approve_below: float = Field(default=0.25, ge=0.0, le=1.0)
    pending_review_below: float = Field(default=0.55, ge=0.0, le=1.0)
    # scores >= pending_review_below are classified as high-risk / auto-rejected


class DecisionThresholds(BaseModel):
    """
    Explicit Phase 5 decision thresholds on the 0–100 ``scam_score`` scale.

    Defaults: score < 30 → clear, 30–70 → review, ≥ 70 → block.
    Easy to tune per deployment without changing scoring code.
    """

    clear_below: float = Field(
        default=30.0,
        ge=0.0,
        le=100.0,
        description="scam_score strictly below this → clear (unless confidence forces review)",
    )
    block_at_or_above: float = Field(
        default=70.0,
        ge=0.0,
        le=100.0,
        description="scam_score at or above this → block",
    )


class ConfidenceConfig(BaseModel):
    """Confidence banding and low-confidence decision override."""

    low_below: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence below this forces decision to at least 'review' — "
            "a low-confidence 'clear' is misleading given Phase 1 unreliable fields."
        ),
    )
    medium_below: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Confidence in [low_below, medium_below) → medium; else high",
    )


class FeatureFlags(BaseModel):
    """Toggle individual feature groups on/off without changing pipeline code."""

    enable_text_features: bool = True
    enable_company_features: bool = True
    enable_url_features: bool = True
    enable_stipend_features: bool = True
    enable_temporal_features: bool = True
    enable_structural_features: bool = True
    enable_ml_risk_engine: bool = False  # off until a trained model is present


class EmbeddingConfig(BaseModel):
    """
    Shared sentence-embedding settings for Prompt 2 (text features) and
    Prompt 5 (duplicate index).  Both modules must read ``sbert_model_name``
    from here — never hardcode the model string in two places.
    """

    sbert_model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description=(
            "sentence-transformers model id used by title_summary_alignment, "
            "boilerplate_similarity, and DuplicateIndex.  Swap to e.g. "
            "all-mpnet-base-v2 for higher quality at higher cost."
        ),
    )


class Config(BaseModel):
    """Top-level config object — instantiate once and share."""

    weights: ScoringWeights = Field(default_factory=ScoringWeights)
    blend_weights: BlendWeights = Field(default_factory=BlendWeights)
    rule_weights: RuleWeights = Field(default_factory=RuleWeights)
    rule_thresholds: RuleThresholds = Field(default_factory=RuleThresholds)
    thresholds: Thresholds = Field(default_factory=Thresholds)
    decision_thresholds: DecisionThresholds = Field(default_factory=DecisionThresholds)
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    flags: FeatureFlags = Field(default_factory=FeatureFlags)

    # When Prompt 6 rule 1 (hard_disqualifying_signals) fires, force this
    # decision regardless of the blended score.  Some deployments prefer
    # human review even on the clearest cases rather than full auto-block.
    hard_disqualifying_decision: Literal["block", "review"] = "block"


# Module-level singleton — override fields as needed in tests or via env vars.
cfg = Config()
