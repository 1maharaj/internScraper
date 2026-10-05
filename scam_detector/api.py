"""
api.py
------
FastAPI production REST API for the iFind Scam Detection Engine.

Endpoints:
  - GET  /health          -> Status, version, and component availability
  - POST /score           -> Score a single internship listing
  - POST /score/batch     -> Batch-score a list of listings with cross-record graph/duplicate detection
  - POST /graph/analyze   -> Network topological metrics and coordinated cluster analysis
  - POST /explain         -> Human review report rendering and feature attribution breakdown
  - POST /feedback        -> Record a moderator approve/reject decision (human-in-the-loop)
  - GET  /feedback/stats  -> Progress toward calibration/supervised-model label thresholds
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

try:
    from fastapi import FastAPI, HTTPException, status
    from fastapi.middleware.cors import CORSMiddleware
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

from scam_detector.config import cfg
from scam_detector.pipeline import score_batch, score_record
from scam_detector.scoring.explain import render_explanation, ScamScoreResult
from scam_detector.scoring.rules_engine import _default_rules
from scam_detector.features.reputation_features import ReputationStore, company_reputation_score
from scam_detector.features.graph_features import (
    build_company_infrastructure_graph,
    compute_graph_network_metrics,
    company_network_risk_profile,
)
from scam_detector.feedback import FeedbackStore, ReviewFeedback, build_training_labels

_CALIBRATION_MIN_SAMPLES = 50
_SUPERVISED_MIN_SAMPLES = 200

log = logging.getLogger("scam_detector.api")


# ---------------------------------------------------------------------------
# Pydantic Request / Response Models
# ---------------------------------------------------------------------------


class InternshipInput(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    title: Optional[str] = Field(default=None, alias="name")
    company: Optional[str] = None
    applyLink: Optional[str] = Field(default=None, alias="apply_link")
    summary: Optional[str] = None
    description: Optional[str] = None
    location: Optional[str] = Field(default=None, alias="city")
    isRemote: Optional[bool] = Field(default=None, alias="is_remote")
    stipend: Optional[Any] = None
    perks: Optional[Any] = None
    skills: Optional[list[str]] = None
    openings: Optional[int] = None
    postedDate: Optional[str] = Field(default=None, alias="datePublished")
    deadline: Optional[str] = Field(default=None, alias="validThrough")
    payment_required: Optional[int] = None
    registration_fee: Optional[float] = None
    fake_certificate_offer: Optional[int] = None
    recruiter_email_type: Optional[str] = None
    suspicious_email_domain: Optional[int] = None
    emotional_manipulation_score: Optional[float] = None
    phishing_language_score: Optional[float] = None


class ScoreResponse(BaseModel):
    id: str
    company: str
    scam_score: float
    decision: str
    confidence: float
    confidence_level: str
    explanation_summary: str
    triggered_rules: list[str]
    top_contributing_features: list[dict[str, Any]]
    shared_infrastructure: bool
    duplicate_cluster_network_size: int


class ScoreBatchRequest(BaseModel):
    records: list[dict[str, Any]] = Field(..., min_length=1)


class BatchScoreResponse(BaseModel):
    total_records: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    results: list[dict[str, Any]]


class GraphAnalyzeRequest(BaseModel):
    records: list[dict[str, Any]] = Field(..., min_length=1)
    target_company: Optional[str] = None


class GraphAnalyzeResponse(BaseModel):
    metrics: dict[str, Any]
    target_company_profile: Optional[dict[str, Any]] = None


class RuleInfo(BaseModel):
    rule_id: str
    description: str
    weight: float


class RulesListResponse(BaseModel):
    total_rules: int
    rules: list[RuleInfo]


class CompanyReputationResponse(BaseModel):
    company: str
    known: bool
    reputation_score: Optional[float] = None
    risk_level: str = "unknown"
    total_postings: int = 0
    clear_count: int = 0
    review_count: int = 0
    block_count: int = 0
    average_scam_score: float = 0.0


class SystemStatsResponse(BaseModel):
    status: str
    version: str
    total_rules: int
    decision_thresholds: dict[str, float]
    rule_weights: dict[str, float]


class BenchmarkSampleRequest(BaseModel):
    records: list[dict[str, Any]] = Field(..., min_length=1)
    label_field: str = "is_fake_posting"
    decision_threshold: float = 50.0


class BenchmarkSampleResponse(BaseModel):
    total_samples: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    latency_ms_per_record: float


class FeedbackRequest(BaseModel):
    """
    A moderator's real-world verdict on a previously-scored listing.

    ``record_id`` must match the same id scam_detector used when scoring
    the listing (its ``_id`` / MongoDB ObjectId string) — this is how a
    future posting from the same company gets linked back to this verdict
    via the reputation store.
    """

    record_id: str = Field(description="The listing's _id, exactly as scored")
    moderator_decision: Literal["approve", "reject"] = Field(
        description="approve = moderator confirms the listing is legitimate; "
        "reject = moderator confirms it is a scam"
    )
    prior_pipeline_decision: Optional[Literal["clear", "review", "block"]] = Field(
        default=None,
        description="What scam_detector originally decided for this listing, "
        "if known — distinguishes an overturned false-positive from a "
        "moderator simply confirming a 'clear' listing.",
    )
    reviewer_notes: str = Field(default="", description="Free-text rationale, e.g. a rejection reason")


class FeedbackResponse(BaseModel):
    record_id: str
    reviewer_decision: str
    recorded_at: str
    total_feedback_count: int
    calibration_ready: bool
    supervised_ready: bool
    message: str


class FeedbackStatsResponse(BaseModel):
    total_labeled: int
    scam_count: int
    legit_count: int
    calibration_min_required: int
    supervised_min_required: int
    calibration_ready: bool
    supervised_ready: bool


# ---------------------------------------------------------------------------
# FastAPI App Factory
# ---------------------------------------------------------------------------


def create_app() -> Any:
    if not _FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI is not installed. Install with: pip install fastapi uvicorn")

    app = FastAPI(
        title="iFind Scam Detection API",
        version="0.1.0",
        description="ML + Deterministic Rules + Graph Infrastructure Scam Detection Engine",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health_check() -> dict[str, Any]:
        from scam_detector.scoring.anomaly_model import _SHAP_AVAILABLE
        return {
            "status": "healthy",
            "version": "0.1.0",
            "shap_available": _SHAP_AVAILABLE,
            "enable_shap_explanations": cfg.flags.enable_shap_anomaly_explanations,
            "decision_thresholds": {
                "clear_below": cfg.decision_thresholds.clear_below,
                "block_at_or_above": cfg.decision_thresholds.block_at_or_above,
            },
        }

    @app.post("/score", response_model=ScoreResponse)
    def score_single(item: InternshipInput) -> ScoreResponse:
        try:
            raw_dict = item.model_dump(by_alias=True, exclude_none=True)
            batch_res = score_batch([raw_dict])
            if not batch_res:
                raise HTTPException(status_code=500, detail="Scoring engine returned empty result")
            scored = batch_res[0]
            rid = str(scored.get("_id") or scored.get("id") or "item_0")
            co = str(scored.get("company") or "Unknown")
            
            return ScoreResponse(
                id=rid,
                company=co,
                scam_score=float(scored.get("scam_score", 0.0)),
                decision=str(scored.get("decision", "clear")),
                confidence=float(scored.get("confidence", 1.0)),
                confidence_level=str(scored.get("confidence_level", "high")),
                explanation_summary=str(scored.get("explanation_summary", "")),
                triggered_rules=list(scored.get("triggered_rules", [])),
                top_contributing_features=list(scored.get("top_contributing_features", [])),
                shared_infrastructure=bool(scored.get("shared_infrastructure", False)),
                duplicate_cluster_network_size=int(scored.get("duplicate_cluster_network_size", 1)),
            )
        except Exception as exc:
            log.exception("Error scoring record: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/score/batch", response_model=BatchScoreResponse)
    def score_batch_endpoint(req: ScoreBatchRequest) -> BatchScoreResponse:
        try:
            scored_records = score_batch(req.records)
            high_count = sum(1 for r in scored_records if r.get("decision") == "block")
            med_count = sum(1 for r in scored_records if r.get("decision") == "review")
            low_count = sum(1 for r in scored_records if r.get("decision") == "clear")

            return BatchScoreResponse(
                total_records=len(scored_records),
                high_risk_count=high_count,
                medium_risk_count=med_count,
                low_risk_count=low_count,
                results=scored_records,
            )
        except Exception as exc:
            log.exception("Error scoring batch: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/graph/analyze", response_model=GraphAnalyzeResponse)
    def analyze_graph_endpoint(req: GraphAnalyzeRequest) -> GraphAnalyzeResponse:
        try:
            graph = build_company_infrastructure_graph(req.records)
            metrics = compute_graph_network_metrics(graph)
            
            profile = None
            if req.target_company:
                profile = company_network_risk_profile(req.target_company, graph)

            return GraphAnalyzeResponse(
                metrics=metrics,
                target_company_profile=profile,
            )
        except Exception as exc:
            log.exception("Error analyzing graph: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/rules", response_model=RulesListResponse)
    def list_rules() -> RulesListResponse:
        rules = _default_rules(cfg)
        rule_list = [
            RuleInfo(
                rule_id=getattr(r, "rule_id", "unknown"),
                description=getattr(r, "__doc__", "").strip().split("\n")[0] if getattr(r, "__doc__", "") else getattr(r, "rule_id", ""),
                weight=float(getattr(cfg.rule_weights, getattr(r, "rule_id", ""), 0.5)),
            )
            for r in rules
        ]
        return RulesListResponse(total_rules=len(rule_list), rules=rule_list)

    @app.get("/reputation/{company_name}", response_model=CompanyReputationResponse)
    def get_company_reputation(company_name: str) -> CompanyReputationResponse:
        rep_store = ReputationStore(cfg.reputation.store_path)
        all_reps = rep_store.get_all_reputations()
        norm_name = company_name.strip().lower()
        if norm_name not in all_reps:
            return CompanyReputationResponse(
                company=company_name,
                known=False,
                reputation_score=None,
                risk_level="unknown",
            )
        rep = all_reps[norm_name]
        score = company_reputation_score(company_name, rep_store)
        risk = "low" if (score or 0.0) < 0.35 else ("medium" if (score or 0.0) < 0.70 else "high")
        return CompanyReputationResponse(
            company=company_name,
            known=True,
            reputation_score=score,
            risk_level=risk,
            total_postings=rep.total_postings,
            clear_count=rep.clear_count,
            review_count=rep.review_count,
            block_count=rep.block_count,
            average_scam_score=rep.average_scam_score,
        )

    @app.get("/stats", response_model=SystemStatsResponse)
    def get_stats() -> SystemStatsResponse:
        rules = _default_rules(cfg)
        return SystemStatsResponse(
            status="healthy",
            version="0.1.0",
            total_rules=len(rules),
            decision_thresholds={
                "clear_below": cfg.decision_thresholds.clear_below,
                "block_at_or_above": cfg.decision_thresholds.block_at_or_above,
            },
            rule_weights=cfg.rule_weights.model_dump(),
        )

    @app.post("/benchmark/sample", response_model=BenchmarkSampleResponse)
    def run_benchmark_sample(req: BenchmarkSampleRequest) -> BenchmarkSampleResponse:
        import time
        if not req.records:
            raise HTTPException(status_code=400, detail="No records provided for benchmark")

        t0 = time.perf_counter()
        scored = score_batch(req.records, config=cfg)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        tp = fp = tn = fn = 0
        for r, sc in zip(req.records, scored):
            y_true = int(r.get(req.label_field, 0))
            score = float(sc.get("scam_score", 0.0))
            y_pred = 1 if score >= req.decision_threshold else 0
            if y_true == 1 and y_pred == 1:
                tp += 1
            elif y_true == 0 and y_pred == 1:
                fp += 1
            elif y_true == 0 and y_pred == 0:
                tn += 1
            else:
                fn += 1

        total = len(req.records)
        acc = (tp + tn) / max(1, total)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = (2 * prec * rec) / max(1e-6, prec + rec) if (prec + rec) > 0 else 0.0

        return BenchmarkSampleResponse(
            total_samples=total,
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            accuracy=round(acc, 4),
            precision=round(prec, 4),
            recall=round(rec, 4),
            f1_score=round(f1, 4),
            latency_ms_per_record=round(elapsed_ms / max(1, total), 2),
        )

    @app.post("/feedback", response_model=FeedbackResponse)
    def record_feedback(req: FeedbackRequest) -> FeedbackResponse:
        """
        Record a moderator's real verdict on a listing (human-in-the-loop).

        This is the mechanism by which the system "learns" over time:
        - Immediately: any FUTURE posting from the same company gets scored
          via ``company_reputation_score``, which checks this feedback store
          for a ``confirmed_scam`` tied to that company's past record_ids and
          returns maximum risk — no retraining needed, works from the very
          first labeled example.
        - Over time: once enough labels accumulate, ``calibration_ready`` /
          ``supervised_ready`` flip to True, at which point the (currently
          dormant) calibration and supervised-model layers in ``config.py``
          can be enabled to generalize beyond company identity.
        """
        try:
            if req.moderator_decision == "reject":
                reviewer_decision = "confirmed_scam"
            elif req.prior_pipeline_decision in ("review", "block"):
                reviewer_decision = "false_positive"
            else:
                reviewer_decision = "confirmed_legit"

            store = FeedbackStore(cfg.feedback.store_path)
            event = store.record_feedback(
                ReviewFeedback(
                    record_id=req.record_id,
                    reviewer_decision=reviewer_decision,
                    reviewer_notes=req.reviewer_notes,
                )
            )

            history = store.load_feedback_history()
            labeled_df = build_training_labels(history)
            total = len(labeled_df)
            scam_count = int((labeled_df["label"] == 1).sum()) if total else 0
            legit_count = total - scam_count
            calibration_ready = total >= _CALIBRATION_MIN_SAMPLES
            supervised_ready = (
                total >= _SUPERVISED_MIN_SAMPLES
                and min(scam_count, legit_count) >= 0.20 * total
            )

            return FeedbackResponse(
                record_id=event.record_id,
                reviewer_decision=event.reviewer_decision,
                recorded_at=event.timestamp.isoformat(),
                total_feedback_count=total,
                calibration_ready=calibration_ready,
                supervised_ready=supervised_ready,
                message=(
                    f"Recorded as '{reviewer_decision}'. "
                    f"{total} labeled examples so far "
                    f"({scam_count} scam, {legit_count} legit)."
                ),
            )
        except Exception as exc:
            log.exception("Error recording feedback: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/feedback/stats", response_model=FeedbackStatsResponse)
    def get_feedback_stats() -> FeedbackStatsResponse:
        store = FeedbackStore(cfg.feedback.store_path)
        history = store.load_feedback_history()
        labeled_df = build_training_labels(history)
        total = len(labeled_df)
        scam_count = int((labeled_df["label"] == 1).sum()) if total else 0
        legit_count = total - scam_count
        calibration_ready = total >= _CALIBRATION_MIN_SAMPLES
        supervised_ready = (
            total >= _SUPERVISED_MIN_SAMPLES
            and min(scam_count, legit_count) >= 0.20 * total
        )
        return FeedbackStatsResponse(
            total_labeled=total,
            scam_count=scam_count,
            legit_count=legit_count,
            calibration_min_required=_CALIBRATION_MIN_SAMPLES,
            supervised_min_required=_SUPERVISED_MIN_SAMPLES,
            calibration_ready=calibration_ready,
            supervised_ready=supervised_ready,
        )

    return app


# Module-level default application instance
if _FASTAPI_AVAILABLE:
    app = create_app()
else:
    app = None
