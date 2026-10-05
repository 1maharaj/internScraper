"""
explain.py
----------
Final :class:`ScamScoreResult` model and human-reviewer report rendering.

``render_explanation`` produces a multi-line report suitable for Phase 6's
Human Review UI.  The legacy ``explain()`` helper remains as a thin bridge
that routes through :class:`~scam_detector.scoring.risk_engine.RiskEngine`.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, computed_field

from scam_detector.scoring.rules_engine import RuleFinding


class RiskLabel(str, Enum):
    """Legacy 3-way label mapped from ``decision`` for older callers."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ScoreContributor(BaseModel):
    """One signal that contributed to the final score (legacy shape)."""

    source: str = Field(description="e.g. 'rules.hard_disqualifying_signals' or 'anomaly'")
    label: str = Field(description="Human-readable signal name")
    contribution: float = Field(ge=0.0, le=1.0)


Decision = Literal["clear", "review", "block"]
ConfidenceLevel = Literal["low", "medium", "high"]


class ScamScoreResult(BaseModel):
    """
    Final Scam Score output for a single internship (Phase 1 / Phase 5).

    ``confidence`` / ``confidence_level`` are kept distinct from ``scam_score``
    so downstream UIs can visually and programmatically distinguish uncertain
    verdicts rather than silently folding confidence into the numeric score.
    """

    scam_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Blended risk score on a 0–100 scale",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="How much to trust the score given data-quality / completeness",
    )
    confidence_level: ConfidenceLevel = Field(
        default="high",
        description="Banded confidence for UI / programmatic gating (low|medium|high)",
    )
    decision: Decision = Field(default="clear")
    triggered_rules: list[str] = Field(default_factory=list)
    top_contributing_features: list[tuple[str, float]] = Field(
        default_factory=list,
        description="(feature_name, contribution) pairs, typically from anomaly explain",
    )
    explanation_method: str = Field(
        default="shap",
        description="Method used for top_contributing_features ('shap' or 'z_score_approximation')",
    )
    explanation_summary: str = Field(
        default="",
        description="Short human-readable sentence summarising the verdict",
    )

    # Extra detail retained for render_explanation / audit (not part of the
    # minimal public contract, but required to list rule explanation strings).
    triggered_rule_findings: list[RuleFinding] = Field(default_factory=list)
    rules_score: float = Field(default=0.0, ge=0.0, le=1.0)
    anomaly_score: float = Field(default=0.0, ge=0.0, le=1.0)
    supervised_score: float | None = Field(default=None, ge=0.0, le=1.0)
    reputation_score: float | None = Field(default=None, ge=0.0, le=1.0)
    hard_disqualifying_forced: bool = Field(
        default=False,
        description="True when decision was forced by hard-disqualifying rule policy",
    )
    low_confidence_forced_review: bool = Field(
        default=False,
        description="True when low confidence upgraded a clear → review",
    )

    # ── Backward-compat aliases used by older pipeline / smoke tests ──────

    @computed_field  # type: ignore[prop-decorator]
    @property
    def final_score(self) -> float:
        """0–1 alias of ``scam_score / 100``."""
        return round(self.scam_score / 100.0, 6)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def label(self) -> RiskLabel:
        mapping = {
            "clear": RiskLabel.LOW,
            "review": RiskLabel.MEDIUM,
            "block": RiskLabel.HIGH,
        }
        return mapping[self.decision]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_hard_reject(self) -> bool:
        return self.decision == "block" and self.hard_disqualifying_forced

    @computed_field  # type: ignore[prop-decorator]
    @property
    def summary(self) -> str:
        return self.explanation_summary

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ml_score(self) -> float:
        return self.anomaly_score

    @computed_field  # type: ignore[prop-decorator]
    @property
    def top_contributors(self) -> list[ScoreContributor]:
        out: list[ScoreContributor] = []
        for finding in self.triggered_rule_findings:
            out.append(
                ScoreContributor(
                    source=f"rules.{finding.rule_id}",
                    label=finding.description,
                    contribution=finding.weight,
                )
            )
        for name, value in self.top_contributing_features[:5]:
            # Contributions from anomaly explain are unbounded z-magnitudes;
            # clamp display contribution into [0, 1] for the legacy model.
            out.append(
                ScoreContributor(
                    source=f"anomaly.{name}",
                    label=name,
                    contribution=min(1.0, max(0.0, float(value) / 10.0)),
                )
            )
        return out


def render_explanation(result: ScamScoreResult) -> str:
    """
    Produce a readable multi-line report for a human reviewer UI (Phase 6).

    Includes every triggered rule with its explanation string, the top 5
    contributing anomaly features, and a confidence caveat when relevant.
    """
    lines: list[str] = [
        "════════════════════════════════════════════════════════════",
        "  SCAM DETECTION — HUMAN REVIEW REPORT",
        "════════════════════════════════════════════════════════════",
        f"  Scam score : {result.scam_score:.1f} / 100",
        f"  Decision   : {result.decision.upper()}",
        (
            f"  Confidence : {result.confidence:.2f}  "
            f"[{result.confidence_level.upper()}]"
        ),
        f"  Rules (0–1): {result.rules_score:.3f}   "
        f"Anomaly (0–1): {result.anomaly_score:.3f}",
    ]
    if result.reputation_score is not None:
        lines.append(f"  Reputation (0–1): {result.reputation_score:.3f}")
    lines.extend([
        "────────────────────────────────────────────────────────────",
        f"  Summary: {result.explanation_summary or '(none)'}",
        "────────────────────────────────────────────────────────────",
        "  Triggered rules:",
    ])

    findings = result.triggered_rule_findings
    if not findings and result.triggered_rules:
        for rid in result.triggered_rules:
            lines.append(f"    • {rid}")
    elif not findings:
        lines.append("    (none)")
    else:
        for finding in findings:
            lines.append(
                f"    • [{finding.rule_id}] weight={finding.weight:.2f} — "
                f"{finding.description}"
            )
            if finding.explanation:
                lines.append(f"        {finding.explanation}")

    lines.append("────────────────────────────────────────────────────────────")
    top5 = list(result.top_contributing_features[:5])
    if result.explanation_method == "shap":
        lines.append("  Top contributing anomaly features (SHAP values):")
        if not top5:
            lines.append("    (none provided)")
        else:
            for i, (name, value) in enumerate(top5, start=1):
                lines.append(f"    {i}. {name}: {value:+.4f} (SHAP)")
    else:
        lines.append("  Top contributing anomaly features (|z-score| approx):")
        if not top5:
            lines.append("    (none provided)")
        else:
            for i, (name, value) in enumerate(top5, start=1):
                lines.append(f"    {i}. {name}: {value:.4f} (z-score)")

    if result.confidence_level == "low" or result.low_confidence_forced_review:
        lines.append("────────────────────────────────────────────────────────────")
        lines.append("  ⚠ CONFIDENCE CAVEAT")
        lines.append(
            "    Score confidence is LOW.  Phase 1 established that several "
            "fields (deadlineDate, degree defaults, company mismapping) are "
            "currently unreliable.  Treat this verdict as provisional and "
            "prefer human review over automated clear/block actions."
        )
        if result.low_confidence_forced_review:
            lines.append(
                "    Note: decision was upgraded from clear → review because "
                "confidence fell below the configured low-confidence threshold."
            )

    if result.hard_disqualifying_forced:
        lines.append("────────────────────────────────────────────────────────────")
        lines.append(
            "  ⚠ Hard-disqualifying rule forced this decision per deployment "
            "policy (independent of the blended numeric score)."
        )

    lines.append("════════════════════════════════════════════════════════════")
    return "\n".join(lines)


def explain(
    rules_result: object,  # RulesResult
    risk_result: object,  # RiskEngineResult (legacy ML stub) or unused
    features: object,  # FeatureVector / record-like
) -> ScamScoreResult:
    """
    Legacy bridge: combine rules + optional ML/anomaly stub into ScamScoreResult.

    Prefer :meth:`RiskEngine.score_record` for new code.
    """
    from scam_detector.scoring.risk_engine import (
        RiskEngine,
        compute_confidence_score,
    )

    anomaly = 0.0
    if risk_result is not None and getattr(risk_result, "model_available", False):
        anomaly = float(getattr(risk_result, "score", 0.0) or 0.0)

    confidence = compute_confidence_score(features)
    return RiskEngine().score_record(
        record=features,
        rule_result=rules_result,  # type: ignore[arg-type]
        anomaly_score=anomaly,
        confidence_score=confidence,
    )
