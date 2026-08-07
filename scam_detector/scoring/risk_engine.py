"""
risk_engine.py
--------------
Capstone scoring layer that ties the deterministic rules engine (Prompt 6)
and the unsupervised anomaly model (Prompt 7) into Phase 1 / Phase 5's
final Scam Score.

Also retains a thin legacy ``RiskEngineResult`` / ``apply_risk_engine`` stub
for an optional supervised ML model artifact (not required for MVP).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from pydantic import BaseModel, Field

from scam_detector.config import Config, cfg as _default_cfg
from scam_detector.scoring.explain import ConfidenceLevel, Decision, ScamScoreResult
from scam_detector.scoring.rules_engine import RuleFinding, RulesResult

_HARD_DISQUALIFYING_RULE_ID = "hard_disqualifying_signals"


# ---------------------------------------------------------------------------
# Legacy supervised-ML stub (optional; not the Phase 5 blend path)
# ---------------------------------------------------------------------------


class RiskEngineResult(BaseModel):
    """Output from an optional supervised ML risk model (legacy stub)."""

    score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Calibrated scam probability from the ML model",
    )
    model_version: str = Field(default="none", description="Identifier of the loaded model")
    model_available: bool = Field(
        default=False,
        description="False when no trained model is present — score defaults to neutral 0.5",
    )


def apply_risk_engine(features: object) -> RiskEngineResult:  # features: FeatureVector
    """
    Run an optional supervised ML model against *features*.

    Returns a neutral (no-model) skeleton until a trained artifact is present
    and ``cfg.flags.enable_ml_risk_engine`` is True.  The Phase 5 blend path
    uses :class:`RiskEngine` with the unsupervised anomaly score instead.
    """
    # TODO: load model from models/, vectorise features, return probability
    return RiskEngineResult()


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


def _get_flags(record: Any) -> Mapping[str, Any]:
    if record is None:
        return {}
    if hasattr(record, "flags") and isinstance(record.flags, Mapping):
        return record.flags
    if isinstance(record, Mapping):
        flags = record.get("flags") or record.get("_flags") or {}
        if isinstance(flags, Mapping):
            return flags
    return {}


def _get_nested(obj: Any, *path: str, default: Any = None) -> Any:
    cur = obj
    for key in path:
        if cur is None:
            return default
        if isinstance(cur, Mapping):
            cur = cur.get(key, default)
        else:
            cur = getattr(cur, key, default)
    return cur if cur is not None else default


def compute_confidence_score(record: Any, features: Any | None = None) -> float:
    """
    Derive a 0–1 confidence score from field completeness and scraper-suspect flags.

    Phase 1 established that deadlineDate, degree defaults, and company
    mismapping are currently unreliable — when those flags fire, confidence
    must drop so a numeric "clear" is never presented as trustworthy.
    """
    src = features if features is not None else record

    completeness = _get_nested(src, "structural", "field_completeness", default=None)
    if completeness is None:
        completeness = _get_nested(src, "structural", "completeness_ratio", default=None)
    if completeness is None and isinstance(record, Mapping):
        completeness = record.get("field_completeness")
    if completeness is None:
        completeness = 0.5  # unknown completeness → mid confidence baseline

    confidence = float(max(0.0, min(1.0, completeness)))

    flags = _get_flags(record)
    # Also honour FeatureVector.company.is_suspect
    company_suspect = bool(flags.get("company_suspect")) or bool(
        _get_nested(src, "company", "is_suspect", default=False)
    )
    degree_suspect = bool(flags.get("degree_suspect_default"))
    deadline_missing = bool(flags.get("deadline_missing"))

    if company_suspect:
        confidence *= 0.55
    if degree_suspect:
        confidence *= 0.75
    if deadline_missing:
        confidence *= 0.85

    # Check if sentence-transformers is installed (SBERT available)
    from scam_detector.features.text_features import _sbert_model
    if _sbert_model() is None:
        # Lower confidence due to missing optional dependency for semantic features
        confidence *= 0.80

    return round(max(0.0, min(1.0, confidence)), 4)


def confidence_level_for(score: float, config: Config | None = None) -> ConfidenceLevel:
    cfg = config or _default_cfg
    if score < cfg.confidence.low_below:
        return "low"
    if score < cfg.confidence.medium_below:
        return "medium"
    return "high"


# ---------------------------------------------------------------------------
# Explanation helpers
# ---------------------------------------------------------------------------


def _build_explanation_summary(
    findings: Sequence[RuleFinding],
    decision: Decision,
    *,
    hard_forced: bool,
    low_conf_forced: bool,
) -> str:
    if hard_forced and findings:
        primary = next(
            (f for f in findings if f.rule_id == _HARD_DISQUALIFYING_RULE_ID),
            findings[0],
        )
        return (
            f"Flagged primarily for {primary.description.lower()} — "
            f"hard-disqualifying policy forces '{decision}'."
        )

    if not findings:
        if low_conf_forced:
            return (
                "No strong risk rules fired, but confidence is too low to clear "
                "this listing automatically; routed to review."
            )
        if decision == "clear":
            return (
                "No significant risk signals detected; listing appears consistent "
                "with normal postings."
            )
        return f"Decision '{decision}' based on blended score without specific rule triggers."

    # Prefer the highest-weight triggered rules for the sentence
    ordered = sorted(findings, key=lambda f: f.weight, reverse=True)
    parts: list[str] = []
    for finding in ordered[:2]:
        snippet = finding.explanation.strip() if finding.explanation else finding.description
        # Keep the summary short — first clause only
        if len(snippet) > 120:
            snippet = snippet[:117].rstrip() + "…"
        parts.append(snippet[0].lower() + snippet[1:] if snippet else finding.rule_id)

    if len(parts) == 1:
        body = parts[0]
    else:
        body = f"{parts[0]} and {parts[1]}"

    prefix = "Flagged primarily for "
    summary = prefix + body
    if not summary.endswith("."):
        summary += "."
    if low_conf_forced:
        summary += " Low confidence forced review rather than clear."
    return summary


# ---------------------------------------------------------------------------
# RiskEngine — Phase 5 capstone
# ---------------------------------------------------------------------------


class RiskEngine:
    """
    Blend rules + anomaly scores into a final ScamScoreResult.

    Parameters
    ----------
    config:
        Optional config override (weights, thresholds, hard-DQ policy).
    """

    def __init__(self, config: Config | None = None) -> None:
        self._cfg = config or _default_cfg

    def score_record(
        self,
        record: Any,
        rule_result: RulesResult,
        anomaly_score: float,
        confidence_score: float,
        *,
        feature_contributions: list[tuple[str, float]] | None = None,
    ) -> ScamScoreResult:
        """
        Produce the final Scam Score for one internship record.

        Parameters
        ----------
        record:
            Raw / remediated record or feature vector (used for context only;
            confidence is taken from ``confidence_score``).
        rule_result:
            Output of the Prompt 6 rules engine.
        anomaly_score:
            Unsupervised anomaly score in ``[0, 1]`` from Prompt 7.
        confidence_score:
            Pre-computed confidence in ``[0, 1]`` (see ``compute_confidence_score``).
        feature_contributions:
            Optional anomaly ``explain()`` pairs ``(feature, magnitude)`` for
            the reviewer report / ``top_contributing_features``.
        """
        _ = record  # reserved for future context-aware blending
        cfg = self._cfg

        rules_score = float(max(0.0, min(1.0, rule_result.combined_score)))
        anomaly = float(max(0.0, min(1.0, anomaly_score)))
        confidence = float(max(0.0, min(1.0, confidence_score)))

        rw = cfg.blend_weights.rules_weight
        aw = cfg.blend_weights.anomaly_weight
        total_w = rw + aw
        if total_w <= 0:
            rw, aw, total_w = 0.60, 0.40, 1.0
        rw, aw = rw / total_w, aw / total_w

        blended_01 = rw * rules_score + aw * anomaly
        scam_score = round(blended_01 * 100.0, 2)

        triggered = list(rule_result.triggered)
        triggered_ids = list(rule_result.triggered_rule_ids) or [
            f.rule_id for f in triggered
        ]
        hard_triggered = _HARD_DISQUALIFYING_RULE_ID in triggered_ids or any(
            f.rule_id == _HARD_DISQUALIFYING_RULE_ID and f.triggered for f in triggered
        )

        hard_forced = False
        low_conf_forced = False

        if hard_triggered:
            decision: Decision = cfg.hard_disqualifying_decision
            hard_forced = True
        else:
            clear_below = cfg.decision_thresholds.clear_below
            block_at = cfg.decision_thresholds.block_at_or_above
            if scam_score >= block_at:
                decision = "block"
            elif scam_score >= clear_below:
                decision = "review"
            else:
                decision = "clear"

        # Low-confidence "clear" is misleading — never present as confident clear
        if confidence < cfg.confidence.low_below and decision == "clear":
            decision = "review"
            low_conf_forced = True

        level = confidence_level_for(confidence, cfg)

        contributions: list[tuple[str, float]]
        if feature_contributions is not None:
            contributions = [
                (str(name), float(val)) for name, val in feature_contributions
            ]
        else:
            # Fall back to triggered rule weights as contributing signals
            contributions = [
                (f.rule_id, float(f.weight))
                for f in sorted(triggered, key=lambda x: x.weight, reverse=True)
            ]

        summary = _build_explanation_summary(
            triggered,
            decision,
            hard_forced=hard_forced,
            low_conf_forced=low_conf_forced,
        )

        return ScamScoreResult(
            scam_score=scam_score,
            confidence=confidence,
            confidence_level=level,
            decision=decision,
            triggered_rules=triggered_ids,
            top_contributing_features=contributions,
            explanation_summary=summary,
            triggered_rule_findings=list(triggered),
            rules_score=rules_score,
            anomaly_score=anomaly,
            hard_disqualifying_forced=hard_forced,
            low_confidence_forced_review=low_conf_forced,
        )
