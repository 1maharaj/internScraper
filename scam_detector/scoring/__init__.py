"""
scam_detector.scoring
=====================
Scoring layer: takes a :class:`FeatureVector` and produces a risk score
plus a human-readable explanation.

Exports
-------
  RiskLabel          — Enum: LOW | MEDIUM | HIGH
  RulesResult        — Output of the deterministic rules engine
  RiskEngine         — Capstone blender (rules + anomaly → ScamScoreResult)
  RiskEngineResult   — Optional supervised ML stub output
  ScamScoreResult    — Final combined result with explanation
  AnomalyModel       — Unsupervised IsolationForest anomaly layer
  TextAnomalyModel   — Phase-2+ embedding-space anomaly stub
  assemble_feature_matrix — Flatten FeatureVectors into a numeric matrix
  compute_confidence_score — Derive confidence from completeness / suspect flags
  apply_rules        — Run deterministic rules
  apply_risk_engine  — Run optional supervised ML stub
  explain            — Legacy bridge into RiskEngine
  render_explanation — Multi-line human-reviewer report
"""

from scam_detector.scoring.rules_engine import RulesResult, apply_rules
from scam_detector.scoring.risk_engine import (
    RiskEngine,
    RiskEngineResult,
    apply_risk_engine,
    compute_confidence_score,
)
from scam_detector.scoring.explain import (
    ScamScoreResult,
    RiskLabel,
    explain,
    render_explanation,
)
from scam_detector.scoring.anomaly_model import (
    AnomalyModel,
    TextAnomalyModel,
    assemble_feature_matrix,
)

__all__: list[str] = [
    "RiskLabel",
    "RulesResult",
    "RiskEngine",
    "RiskEngineResult",
    "ScamScoreResult",
    "AnomalyModel",
    "TextAnomalyModel",
    "assemble_feature_matrix",
    "compute_confidence_score",
    "apply_rules",
    "apply_risk_engine",
    "explain",
    "render_explanation",
]
