"""
pipeline.py
-----------
Phase 6 batch orchestrator — ties Prompts 1–8 into a single end-to-end run.

Architecture
------------
    load raw records (format_internship output schema)
      → data_quality.remediate_batch
      → corpus structures once:
            DuplicateIndex.build  (or lexical fallback)
            peer groups for stipend / openings z-scores
            company posting-frequency / burst lookups
      → per-record feature extraction (text, company/URL, stipend, temporal, structural)
      → assemble_feature_matrix → AnomalyModel.fit / score
      → per-record RulesEngine.run
      → per-record RiskEngine.score_record
      → write JSON array (input order): original + scam_score + decision
            + explanation_summary + confidence

CLI (mirrors internScraper orchestrator style)
---------------------------------------------
    python -m scam_detector.pipeline <input.json> <output.json> [--sample N]

Usage (library)
---------------
    from scam_detector.pipeline import run_pipeline, process_records
    run_pipeline("internships.json", "scored.json")
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scam_detector.config import Config, cfg as default_cfg
from scam_detector.data_quality.remediate import (
    RemediatedRecord,
    remediate_batch,
    remediate_record,
)
from scam_detector.features import (
    FeatureVector,
    StipendFeatures,
    StructuralFeatures,
    TemporalFeatures,
    extract_all,
    extract_company_url_features,
    extract_text_features,
)
from scam_detector.features.duplicate_detection import (
    DuplicateIndex,
    _record_id,
    _record_text,
    cross_company_duplicate_flag,
)
from scam_detector.features.stipend_features import (
    normalize_stipend_to_hourly_inr,
    stipend_perk_consistency_check,
    stipend_zscore,
)
from scam_detector.features.structural_features import (
    field_completeness_score,
    openings_zscore,
)
from scam_detector.features.temporal_features import (
    deadline_urgency_score,
    posting_burst_score,
)
from scam_detector.scoring.anomaly_model import AnomalyModel, assemble_feature_matrix
from scam_detector.scoring.explain import ScamScoreResult
from scam_detector.scoring.risk_engine import RiskEngine, compute_confidence_score
from scam_detector.scoring.rules_engine import RuleInput, RulesEngine

log = logging.getLogger("scam_detector.pipeline")

# Lexical near-duplicate threshold used when SBERT / DuplicateIndex is unavailable.
_LEXICAL_DUP_THRESHOLD = 0.85
_PEER_JACCARD_MIN = 0.20


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_records(input_path: str | Path) -> list[dict[str, Any]]:
    """
    Load internship records from a JSON file.

    Accepts a bare JSON array or an object with an ``internships`` /
    ``records`` / ``data`` key (common scraper checkpoint shapes).
    """
    path = Path(input_path)
    with path.open(encoding="utf-8") as fh:
        payload = json.load(fh)

    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        for key in ("internships", "records", "data", "items"):
            if isinstance(payload.get(key), list):
                records = payload[key]
                break
        else:
            raise ValueError(
                f"Unsupported JSON object in {path}: expected a list or "
                "an object with an 'internships'/'records' array"
            )
    else:
        raise ValueError(f"Unsupported JSON root type in {path}: {type(payload)}")

    if not all(isinstance(r, dict) for r in records):
        raise ValueError(f"All records in {path} must be JSON objects")
    return records


def write_records(output_path: str | Path, records: list[dict[str, Any]]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False, default=str)
        fh.write("\n")


# ---------------------------------------------------------------------------
# Corpus-level structures
# ---------------------------------------------------------------------------


def _as_str_set(value: Any) -> set[str]:
    if not value:
        return set()
    if isinstance(value, str):
        return {value.strip().lower()} if value.strip() else set()
    if isinstance(value, list):
        return {str(v).strip().lower() for v in value if v and str(v).strip()}
    return set()


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def build_peer_group(
    record: dict[str, Any],
    all_records: list[dict[str, Any]],
    *,
    min_peer_group_size: int = 2,
) -> list[dict[str, Any]]:
    """
    Peer group for stipend / openings z-scores conditioned in priority order:
    1. Subcategory + Remote status + City tier (if city tier is reliable/known)
    2. Subcategory + Remote status
    3. Subcategory
    4. Broad category + Remote status
    5. Broad category

    Returns [] if no priority level yields >= min_peer_group_size records.
    """
    from scam_detector.features.stipend_features import (
        get_role_subcategory,
        get_broad_category,
        get_remote_status,
        get_city_tier,
    )

    subcat = get_role_subcategory(record)
    broad_cat = get_broad_category(subcat)
    remote_st = get_remote_status(record)
    city_t = get_city_tier(record)

    # 1. Subcategory + Remote status + City tier
    if city_t not in ("unknown", ""):
        p1 = [
            r for r in all_records
            if get_role_subcategory(r) == subcat
            and get_remote_status(r) == remote_st
            and get_city_tier(r) == city_t
        ]
        if len(p1) >= min_peer_group_size:
            return p1

    # 2. Subcategory + Remote status
    p2 = [
        r for r in all_records
        if get_role_subcategory(r) == subcat
        and get_remote_status(r) == remote_st
    ]
    if len(p2) >= min_peer_group_size:
        return p2

    # 3. Subcategory alone
    p3 = [
        r for r in all_records
        if get_role_subcategory(r) == subcat
    ]
    if len(p3) >= min_peer_group_size:
        return p3

    # 4. Broad category + Remote status
    p4 = [
        r for r in all_records
        if get_broad_category(get_role_subcategory(r)) == broad_cat
        and get_remote_status(r) == remote_st
    ]
    if len(p4) >= min_peer_group_size:
        return p4

    # 5. Broad category alone
    p5 = [
        r for r in all_records
        if get_broad_category(get_role_subcategory(r)) == broad_cat
    ]
    if len(p5) >= min_peer_group_size:
        return p5

    return []



def _index_by_company(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    by_company: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        key = (rec.get("company") or "").strip().lower()
        by_company[key].append(rec)
    return by_company


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _lexical_near_duplicates(
    records: list[dict[str, Any]],
    *,
    threshold: float = _LEXICAL_DUP_THRESHOLD,
) -> dict[str, list[tuple[str, float]]]:
    """
    O(N²) Jaccard fallback when sentence-transformers / DuplicateIndex is
    unavailable.  Fine for development samples and CI fixtures.
    """
    ids = [_record_id(r, i) for i, r in enumerate(records)]
    tokens = [_token_set(_record_text(r)) for r in records]
    neighbors: dict[str, list[tuple[str, float]]] = {rid: [] for rid in ids}

    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            a, b = tokens[i], tokens[j]
            if not a or not b:
                continue
            sim = len(a & b) / len(a | b)
            if sim >= threshold:
                neighbors[ids[i]].append((ids[j], round(sim, 4)))
                neighbors[ids[j]].append((ids[i], round(sim, 4)))

    for rid in neighbors:
        neighbors[rid].sort(key=lambda x: x[1], reverse=True)
    return neighbors


def _build_duplicate_neighbors(
    records: list[dict[str, Any]],
) -> dict[str, list[tuple[str, float]]]:
    """Prefer DuplicateIndex (Prompt 5); fall back to lexical Jaccard."""
    try:
        index = DuplicateIndex()
        index.build(records)
        neighbors: dict[str, list[tuple[str, float]]] = {}
        for i, rec in enumerate(records):
            rid = _record_id(rec, i)
            neighbors[rid] = index.find_near_duplicates(rid, threshold=0.92)
        log.info(
            "DuplicateIndex built for %d records (SBERT near-duplicate search)",
            len(records),
        )
        return neighbors
    except Exception as exc:
        log.warning(
            "DuplicateIndex unavailable (%s) — using lexical Jaccard fallback",
            exc,
        )
        return _lexical_near_duplicates(records)


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------


def _enrich_feature_vector(
    remediated: RemediatedRecord,
    *,
    all_records: list[dict[str, Any]],
    peer_group: list[dict[str, Any]],
    company_records: list[dict[str, Any]],
    scam_embeddings: Any | None = None,
    min_peer_group_size: int = 8,
    config: Config | None = None,
) -> FeatureVector:
    """Per-record feature extraction using pre-built corpus structures."""
    cfg = config or default_cfg
    raw = remediated.record
    flags = remediated.flags
    raw_with_flags = {**raw, "_flags": flags}

    text = extract_text_features(raw_with_flags, scam_embeddings=scam_embeddings)
    cu = extract_company_url_features(
        raw_with_flags, all_records=all_records, flags=flags
    )

    hourly = normalize_stipend_to_hourly_inr(
        raw.get("stipend") or {},
        raw.get("duration") or {},
    )
    peer_z = stipend_zscore(raw, peer_group, min_peer_group_size=min_peer_group_size)
    contradiction = stipend_perk_consistency_check(raw)
    stipend_val = raw.get("stipend")
    if isinstance(stipend_val, str):
        stipend_type = stipend_val.strip().lower()
    elif isinstance(stipend_val, dict):
        stipend_type = str(stipend_val.get("type") or "unknown")
    else:
        stipend_type = "unknown"

    stipend = StipendFeatures(
        peer_zscore=peer_z,
        hourly_inr=hourly,
        perk_consistency_ok=not contradiction,
        stipend_type=stipend_type,
        is_outlier_high=bool(peer_z is not None and peer_z > cfg.rule_thresholds.stipend_zscore_threshold),
        is_outlier_low=bool(peer_z is not None and peer_z < -cfg.rule_thresholds.stipend_zscore_threshold),
        amount_plausibility_score=1.0 if hourly is not None else 0.5,
        missing_stipend_for_role=hourly is None,
    )

    burst = posting_burst_score(raw, company_records)
    from scam_detector.features.temporal_features import recruiter_posting_velocity
    v24 = recruiter_posting_velocity(raw, all_records, hours=24)
    v72 = recruiter_posting_velocity(raw, all_records, hours=72)

    temporal = TemporalFeatures(
        posting_burst_count=int(burst.get("burst_count") or 0),
        posting_burst_cadence=burst.get("cadence_days"),
        deadline_urgency_score=deadline_urgency_score(raw),
        recruiter_posting_velocity_24h=v24,
        recruiter_posting_velocity_72h=v72,
    )

    completeness = field_completeness_score(raw)
    oz = openings_zscore(raw, peer_group, min_peer_group_size=min_peer_group_size)
    skills = raw.get("skills") or []
    skills_count = len(skills) if isinstance(skills, list) else 0
    responsibilities = raw.get("responsibilities") or []
    resp_count = len(responsibilities) if isinstance(responsibilities, list) else 0

    structural = StructuralFeatures(
        openings_zscore=oz,
        field_completeness=completeness,
        completeness_ratio=completeness,
        skills_count=skills_count,
        skills_count_anomaly=skills_count < 1 or skills_count > 30,
        responsibilities_count=resp_count,
    )

    return FeatureVector(
        text=text,
        company=cu.company,
        url=cu.url,
        stipend=stipend,
        temporal=temporal,
        structural=structural,
    )



def feature_vector_to_rule_input(
    fv: FeatureVector,
    *,
    cross_company_duplicate: bool,
    shared_infrastructure: bool = False,
    ngo_stipend_network: bool = False,
    ngo_stipend_network_company_count: int = 0,
    flags: dict[str, Any] | None = None,
    record: dict[str, Any] | None = None,
) -> RuleInput:
    """Explicit bridge from FeatureVector → RuleInput (avoids duck-type gaps)."""
    rec = record or {}

    # Extract payment / fee indicators
    payment_req = bool(rec.get("payment_required", 0))
    reg_fee = float(rec.get("registration_fee", 0.0) or 0.0)
    fake_cert = bool(rec.get("fake_certificate_offer", 0))

    # Also detect fee phrases from text if not explicitly labeled
    summary_text = str(rec.get("summary") or "") + " " + str(rec.get("description") or "")
    if not payment_req and re.search(r"(?:registration|security|application|training|onboarding)\s+(?:fee|deposit|charge|amount)", summary_text, re.IGNORECASE):
        payment_req = True

    # Recruiter email signals
    recruiter_type = str(rec.get("recruiter_email_type") or "Corporate")
    suspicious_email = bool(rec.get("suspicious_email_domain", 0))
    apply_link = str(rec.get("applyLink") or rec.get("apply_link") or "")
    if "@" in apply_link:
        if any(free_domain in apply_link.lower() for free_domain in ("@gmail.", "@yahoo.", "@hotmail.", "@outlook.", "@proton.")):
            recruiter_type = "Free"

    # Psychological manipulation signals
    emo_score = float(rec.get("emotional_manipulation_score", 0.0) or 0.0)
    phish_score = float(rec.get("phishing_language_score", 0.0) or 0.0)
    urg_score = float(rec.get("urgency_score", fv.text.urgency_score) or 0.0)

    return RuleInput(
        sensitive_info_requested=fv.text.sensitive_info_requested,
        guaranteed_outcome_claim=fv.text.guaranteed_outcome_claim,
        external_form_detected=fv.text.external_form_detected,
        personal_contact_handoff_detected=fv.text.personal_contact_handoff_detected,
        training_program_disguised_as_internship=fv.text.training_program_disguised_as_internship,
        zero_shot_scam_category=fv.text.zero_shot_scam_category,
        zero_shot_scam_confidence=fv.text.zero_shot_scam_confidence,
        scam_corpus_similarity=fv.text.scam_corpus_similarity,
        urgency_score=urg_score,
        genericity_score=fv.text.genericity_score,
        caps_ratio=fv.text.caps_ratio,
        exclamation_count=fv.text.exclamation_count,
        has_repeated_punctuation=fv.text.has_repeated_punctuation,
        summary_truncated=fv.text.summary_truncated,
        company_is_suspect=fv.company.is_suspect,
        typosquat_min_distance=fv.company.typosquat_min_distance,
        domain_age_days=fv.company.domain_age_days,
        is_platform_internal=fv.url.is_platform_internal,
        is_url_shortener=fv.url.is_url_shortener,
        is_known_ats=fv.url.is_known_ats,
        domain_company_similarity=fv.url.domain_company_similarity,
        stipend_peer_zscore=fv.stipend.peer_zscore,
        perk_consistency_ok=fv.stipend.perk_consistency_ok,
        openings_zscore=fv.structural.openings_zscore,
        field_completeness=fv.structural.field_completeness,
        cross_company_duplicate=cross_company_duplicate,
        shared_infrastructure=shared_infrastructure,
        ngo_stipend_network=ngo_stipend_network,
        ngo_stipend_network_company_count=ngo_stipend_network_company_count,
        payment_required=payment_req,
        registration_fee=reg_fee,
        fake_certificate_offer=fake_cert,
        recruiter_email_type=recruiter_type,
        suspicious_email_domain=suspicious_email,
        emotional_manipulation_score=emo_score,
        phishing_language_score=phish_score,
        remediation_flags=dict(flags or {}),
    )


# ---------------------------------------------------------------------------
# Core batch processing
# ---------------------------------------------------------------------------


def process_records(
    raw_records: list[dict[str, Any]],
    *,
    config: Config | None = None,
) -> list[dict[str, Any]]:
    """
    Run the full Phase 6 pipeline on an in-memory list of raw records.

    Returns a new list (same order) where each element is the original record
    plus ``scam_score``, ``decision``, ``explanation_summary``, and
    ``confidence``.
    """
    if not raw_records:
        return []

    cfg = config or default_cfg
    from scam_detector.scoring.risk_engine import clear_baselines_cache
    clear_baselines_cache()
    n = len(raw_records)
    log.info("Processing %d records", n)

    # ── Prompt 1: remediate ───────────────────────────────────────────────
    remediated: list[RemediatedRecord] = remediate_batch(raw_records)
    records = [r.record for r in remediated]
    flags_list = [r.flags for r in remediated]

    # Ensure stable ids for duplicate / matrix indexing
    for i, rec in enumerate(records):
        if not any(rec.get(k) for k in ("_id", "id", "internship_id")):
            rec = {**rec, "_id": f"pipeline-row-{i}"}
        # Add flags to record so downstream steps can access flags easily
        rec = {**rec, "_flags": flags_list[i]}

        # Precompute subcategory and city tier to avoid O(N^2) evaluation overhead
        from scam_detector.features.stipend_features import get_role_subcategory, get_city_tier
        rec["role_subcategory"] = get_role_subcategory(rec)
        rec["city_tier"] = get_city_tier(rec)

        records[i] = rec
        remediated[i] = RemediatedRecord(record=rec, flags=flags_list[i])

    # ── Corpus structures (once) ──────────────────────────────────────────
    min_peer_size = cfg.rule_thresholds.min_peer_group_size
    neighbors_by_id = _build_duplicate_neighbors(records)
    by_company = _index_by_company(records)

    # Build company infrastructure graph
    from scam_detector.features.graph_features import (
        build_company_infrastructure_graph,
        shared_infrastructure_flag,
        duplicate_cluster_network_size,
    )
    infra_graph = build_company_infrastructure_graph(records, neighbors_by_id)
    peer_cache: list[list[dict[str, Any]]] = [
        build_peer_group(rec, records, min_peer_group_size=min_peer_size) for rec in records
    ]

    from scam_detector.features.company_features import (
        is_ngo_or_fundraising_sector,
        build_lump_sum_stipend_index,
        ngo_stipend_network_flag,
    )
    stipend_index = build_lump_sum_stipend_index(
        [(rem.record, rem.flags) for rem in remediated]
    )
    ngo_min_companies = cfg.rule_thresholds.ngo_stipend_min_distinct_companies

    # ── Per-record features (Prompts 2–5) ─────────────────────────────────
    from scam_detector.features.text_features import get_scam_corpus_embeddings
    from scam_detector.feedback import FeedbackStore
    from scam_detector.features.reputation_features import ReputationStore, company_reputation_score

    feedback_store = FeedbackStore(cfg.feedback.store_path)
    rep_store = ReputationStore(cfg.reputation.store_path)

    try:
        scam_embeddings = get_scam_corpus_embeddings(feedback_store, records)
    except Exception:
        scam_embeddings = None

    feature_vectors: list[FeatureVector] = []
    cross_company_flags: list[bool] = []
    shared_infra_flags: list[bool] = []
    cluster_network_sizes: list[int] = []
    ngo_stipend_flags: list[bool] = []
    ngo_stipend_counts: list[int] = []

    for i, rem in enumerate(remediated):
        rec = rem.record
        rid = _record_id(rec, i)
        company_key = (rec.get("company") or "").strip().lower()
        company_recs = by_company.get(company_key, [rec])

        fv = _enrich_feature_vector(
            rem,
            all_records=records,
            peer_group=peer_cache[i],
            company_records=company_recs,
            scam_embeddings=scam_embeddings,
            min_peer_group_size=min_peer_size,
            config=cfg,
        )
        feature_vectors.append(fv)


        dup_neighbors = neighbors_by_id.get(rid, [])
        cross_company_flags.append(
            cross_company_duplicate_flag(rec, dup_neighbors, records)
        )

        company_name = rec.get("company") or ""
        shared_infra_flags.append(
            shared_infrastructure_flag(company_name, infra_graph)
        )
        cluster_network_sizes.append(
            duplicate_cluster_network_size(company_name, infra_graph)
        )

        stipend_flagged, stipend_company_count = ngo_stipend_network_flag(
            rec, stipend_index, min_distinct_companies=ngo_min_companies
        )
        is_ngo = is_ngo_or_fundraising_sector(rec)
        ngo_stipend_flags.append(bool(is_ngo and stipend_flagged))
        ngo_stipend_counts.append(stipend_company_count if is_ngo else 0)

    # ── Prompt 7: anomaly model ───────────────────────────────────────────
    matrix = assemble_feature_matrix(records, feature_vectors)
    anomaly_scores = [0.0] * n
    anomaly_explanations: list[list[tuple[str, float]]] = [[] for _ in range(n)]

    if n >= 2 and not matrix.empty:
        try:
            model = AnomalyModel(random_state=42, contamination="auto", config=cfg)
            model.fit(matrix)
            scored = model.score(matrix)
            anomaly_scores = [float(s) for s in scored]
            anomaly_explanations = model.explain_batch(matrix)
            log.info("AnomalyModel fitted on %d rows", n)
        except Exception as exc:
            log.warning("AnomalyModel failed (%s) — anomaly scores default to 0.0", exc)
    else:
        log.info("Skipping AnomalyModel (need ≥2 records); anomaly scores = 0.0")

    # ── Supervised model (if trained artifact is present) ─────────────────
    from scam_detector.scoring.supervised_model import SupervisedScamModel
    supervised_scores: list[float | None] = [None] * n
    sup_model = SupervisedScamModel(config=cfg)
    if sup_model.load():
        try:
            sup_probas = sup_model.predict_proba(matrix)
            if len(sup_probas) == n:
                supervised_scores = [float(p) for p in sup_probas]
                log.info("SupervisedScamModel evaluated on %d records", n)
        except Exception as exc:
            log.warning("SupervisedScamModel failed (%s) — supervised scores default to None", exc)

    # ── Prompts 6 + 8: rules + risk engine ────────────────────────────────
    rules_engine = RulesEngine(config=cfg)
    risk_engine = RiskEngine(config=cfg)
    outputs: list[dict[str, Any]] = []
    decisions: list[str] = []

    for i, raw in enumerate(raw_records):
        fv = feature_vectors[i]
        rem = remediated[i]
        rule_input = feature_vector_to_rule_input(
            fv,
            cross_company_duplicate=cross_company_flags[i],
            shared_infrastructure=shared_infra_flags[i],
            ngo_stipend_network=ngo_stipend_flags[i],
            ngo_stipend_network_company_count=ngo_stipend_counts[i],
            flags=rem.flags,
            record=raw,
        )
        rule_result = rules_engine.run(rule_input)
        confidence = compute_confidence_score(rem, fv, config=cfg)

        company_name = raw.get("company") or ""
        if fv.company.is_suspect or not company_name.strip():
            rep_score = None
        else:
            rep_score = company_reputation_score(company_name, rep_store, feedback_store)

        result: ScamScoreResult = risk_engine.score_record(
            record=rem,
            rule_result=rule_result,
            anomaly_score=anomaly_scores[i],
            confidence_score=confidence,
            supervised_score=supervised_scores[i],
            reputation_score=rep_score,
            feature_contributions=anomaly_explanations[i],
            explanation_method="shap" if cfg.anomaly.enable_shap and cfg.flags.enable_shap_anomaly_explanations else "z_score_approximation",
        )

        # Build canonical Moderation object matching ifind -> types -> internship.ts
        triggered_flags = [f.explanation or f.description or f.rule_id for f in rule_result.triggered]
        if not triggered_flags and result.explanation_summary:
            triggered_flags = [result.explanation_summary]

        moderation_status = (
            "auto_approved" if result.decision == "clear" else
            "auto_rejected" if result.decision == "block" else
            "pending_review"
        )

        out = dict(raw)
        out["scam_score"] = result.scam_score
        out["decision"] = result.decision
        out["explanation_summary"] = result.explanation_summary
        out["confidence"] = result.confidence
        out["moderation"] = {
            "status": moderation_status,
            "score": result.scam_score,
            "flags": triggered_flags,
            "source": raw.get("source") or "web_scraping",
            "reviewedBy": None,
            "reviewedAt": datetime.now(timezone.utc).isoformat(),
            "rejectionReason": result.explanation_summary if result.decision == "block" else None,
        }
        out["confidence_level"] = result.confidence_level
        out["triggered_rules"] = result.triggered_rules
        out["top_contributing_features"] = [
            {"feature": name, "contribution": contribution}
            for name, contribution in result.top_contributing_features
        ]
        out["risk_breakdown"] = {
            "rules_score": result.rules_score,
            "anomaly_score": result.anomaly_score,
            "supervised_score": result.supervised_score,
            "reputation_score": result.reputation_score,
        }
        out["hard_disqualifying_forced"] = result.hard_disqualifying_forced
        out["low_confidence_forced_review"] = result.low_confidence_forced_review
        out["shared_infrastructure"] = shared_infra_flags[i]
        out["duplicate_cluster_network_size"] = cluster_network_sizes[i]
        outputs.append(out)
        decisions.append(result.decision)

    bucket_counts = Counter(decisions)
    log.info(
        "Decision buckets — clear=%d review=%d block=%d",
        bucket_counts.get("clear", 0),
        bucket_counts.get("review", 0),
        bucket_counts.get("block", 0),
    )

    # ── Update reputation store ───────────────────────────────────────────
    try:
        update_reputations_after_run(remediated, outputs, rep_store)
    except Exception as exc:
        log.warning("Failed to update reputation store: %s", exc)

    # ── Update per-source baselines ────────────────────────────────────────
    if cfg.confidence.enable_source_conditioning:
        try:
            update_source_baselines(raw_records, feature_vectors, outputs, cfg.confidence.source_baseline_path)
        except Exception as exc:
            log.warning("Failed to update source baselines: %s", exc)

    return outputs


def update_source_baselines(
    raw_records: list[dict[str, Any]],
    feature_vectors: list[FeatureVector],
    outputs: list[dict[str, Any]],
    path: str,
) -> None:
    import statistics
    from scam_detector.scoring.risk_engine import clear_baselines_cache

    p = Path(path)
    baselines = {}
    if p.exists():
        try:
            with p.open("r", encoding="utf-8") as fh:
                baselines = json.load(fh)
        except Exception:
            pass

    # Group records by source
    by_source = defaultdict(list)
    for i, raw in enumerate(raw_records):
        source = raw.get("source") or "unknown"
        completeness = feature_vectors[i].structural.field_completeness
        scam_score = outputs[i]["scam_score"]
        by_source[source].append((completeness, scam_score))

    for source, data in by_source.items():
        comp_list = [d[0] for d in data]
        score_list = [d[1] for d in data]

        source_data = baselines.get(
            source,
            {
                "count": 0,
                "mean_completeness": 0.0,
                "mean_scam_score": 0.0,
                "recent_completeness": [],
                "recent_scam_scores": [],
            },
        )

        # Update lists
        source_data["recent_completeness"].extend(comp_list)
        source_data["recent_scam_scores"].extend(score_list)

        # Cap rolling history at 5000 items
        if len(source_data["recent_completeness"]) > 5000:
            source_data["recent_completeness"] = source_data["recent_completeness"][-5000:]
        if len(source_data["recent_scam_scores"]) > 5000:
            source_data["recent_scam_scores"] = source_data["recent_scam_scores"][-5000:]

        source_data["count"] = source_data["count"] + len(comp_list)
        source_data["mean_completeness"] = round(statistics.mean(source_data["recent_completeness"]), 4)
        source_data["mean_scam_score"] = round(statistics.mean(source_data["recent_scam_scores"]), 4)

        baselines[source] = source_data

    # Save to disk
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            json.dump(baselines, fh, indent=2)
    except Exception:
        pass

    # Invalidate cache
    clear_baselines_cache()


def update_reputations_after_run(
    remediated_records: list[RemediatedRecord],
    outputs: list[dict[str, Any]],
    reputation_store: ReputationStore,
) -> None:
    """Update company reputation history with the finalized decisions and scores from this run."""
    from datetime import date, datetime, timezone
    from scam_detector.features.reputation_features import CompanyReputation
    from scam_detector.features.company_features import is_company_suspect, _parse_date
    from scam_detector.features.duplicate_detection import _record_id

    # 1. Load existing reputations
    existing = reputation_store.get_all_reputations()

    # 2. Group updates by company key
    updates_by_company: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for i, rem in enumerate(remediated_records):
        rec = rem.record
        out = outputs[i]
        company = rec.get("company") or ""
        company_key = company.strip().lower()
        suspect = is_company_suspect(rem.flags)

        if suspect or not company_key:
            continue

        rid = _record_id(rec, i)
        date_published = rec.get("datePublished") or rec.get("date_published") or ""
        updates_by_company[company_key].append({
            "record_id": rid,
            "decision": out.get("decision", "review"),
            "scam_score": out.get("scam_score", 50.0),
            "date": date_published,
        })

    if not updates_by_company:
        return

    # 3. Create or update CompanyReputation for each company
    updated_records: list[CompanyReputation] = []
    for company_key, items in updates_by_company.items():
        dates = [d for item in items if (d := _parse_date(item["date"]))]
        min_date_in_run = min(dates).isoformat() if dates else date.today().isoformat()

        if company_key in existing:
            rep = existing[company_key]
            old_total = rep.total_postings
            new_total = old_total + len(items)

            # update totals and counts
            rep.total_postings = new_total
            for item in items:
                dec = item["decision"]
                if dec == "clear":
                    rep.clear_count += 1
                elif dec == "block":
                    rep.block_count += 1
                else:
                    rep.review_count += 1

                if item["record_id"] not in rep.record_ids:
                    rep.record_ids.append(item["record_id"])

            # update average scam score
            sum_scam_score_in_run = sum(item["scam_score"] for item in items)
            rep.average_scam_score = (rep.average_scam_score * old_total + sum_scam_score_in_run) / new_total

            # update first_seen if we found an older one
            if min_date_in_run < rep.first_seen:
                rep.first_seen = min_date_in_run

            rep.last_updated = datetime.now(timezone.utc)
            updated_records.append(rep)
        else:
            clear_count = sum(1 for item in items if item["decision"] == "clear")
            block_count = sum(1 for item in items if item["decision"] == "block")
            review_count = sum(1 for item in items if item["decision"] == "review")
            avg_scam_score = sum(item["scam_score"] for item in items) / len(items)
            record_ids = [item["record_id"] for item in items]

            rep = CompanyReputation(
                company=company_key,
                first_seen=min_date_in_run,
                total_postings=len(items),
                clear_count=clear_count,
                review_count=review_count,
                block_count=block_count,
                average_scam_score=avg_scam_score,
                record_ids=record_ids,
            )
            updated_records.append(rep)

    # 4. Save updates to ReputationStore
    reputation_store.update_reputations(updated_records)


def run_pipeline(
    input_path: str,
    output_path: str,
    *,
    sample: int | None = None,
    config: Config | None = None,
    seed: int = 42,
) -> None:
    """
    Load → score → write.  Optional ``sample`` draws a random subsample for
    fast iteration during development.
    """
    t0 = time.perf_counter()
    records = load_records(input_path)
    log.info("Loaded %d records from %s", len(records), input_path)

    if sample is not None:
        if sample < 0:
            raise ValueError("--sample must be >= 0")
        if sample < len(records):
            rng = random.Random(seed)
            records = rng.sample(records, sample)
            log.info("Subsampled to %d records (--sample %d, seed=%d)", len(records), sample, seed)

    outputs = process_records(records, config=config)
    write_records(output_path, outputs)

    elapsed = time.perf_counter() - t0
    buckets = Counter(r.get("decision", "?") for r in outputs)
    log.info(
        "Wrote %d records to %s in %.2fs | clear=%d review=%d block=%d",
        len(outputs),
        output_path,
        elapsed,
        buckets.get("clear", 0),
        buckets.get("review", 0),
        buckets.get("block", 0),
    )


# ---------------------------------------------------------------------------
# High-Level Scoring APIs
# ---------------------------------------------------------------------------


def score_batch(
    records: list[dict[str, Any]],
    *,
    config: Config | None = None,
) -> list[dict[str, Any]]:
    """
    Score a batch of raw internship records through the full pipeline.

    Alias for :func:`process_records`, applying data remediation, duplicate
    and graph analysis, anomaly scoring, rules engine, and risk aggregation.
    """
    return process_records(records, config=config)


def score_record(
    record: dict[str, Any],
    *,
    config: Config | None = None,
) -> dict[str, Any]:
    """
    Score a single raw internship record.

    Returns the scored record dictionary containing 'scam_score', 'decision',
    'confidence', and 'explanation_summary'.
    """
    results = process_records([record], config=config)
    return results[0] if results else {}


# ---------------------------------------------------------------------------
# Single-record wrapper (backward compatible)
# ---------------------------------------------------------------------------


class ScamDetectorPipeline:
    """
    Stateless single-record wrapper.

    Prefer :func:`run_pipeline` / :func:`process_records` for batch scoring;
    this class remains for ad-hoc single-dict calls and older imports.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.config: Config = config or default_cfg

    def run(
        self,
        raw: dict[str, Any],
        *,
        anomaly_score: float = 0.0,
        feature_contributions: list[tuple[str, float]] | None = None,
    ) -> ScamScoreResult:
        """Score one raw internship dict (no corpus-level anomaly fit)."""
        rem = remediate_record(raw)
        features = extract_all(rem)
        # Minimal RuleInput via batch helper fields available on FeatureVector
        rule_input = feature_vector_to_rule_input(
            features,
            cross_company_duplicate=False,
            flags=rem.flags,
            record=raw,
        )
        rule_result = RulesEngine(config=self.config).run(rule_input)
        confidence = compute_confidence_score(rem, features)
        return RiskEngine(self.config).score_record(
            record=rem,
            rule_result=rule_result,
            anomaly_score=anomaly_score,
            confidence_score=confidence,
            feature_contributions=feature_contributions,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scam_detector.pipeline",
        description=(
            "Run the iFind scam-detection pipeline on a JSON internship file "
            "and write scored results."
        ),
    )
    parser.add_argument(
        "input_json",
        help="Path to input JSON (array of internship records, or object with 'internships')",
    )
    parser.add_argument(
        "output_json",
        help="Path to write scored JSON array",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Randomly subsample N records for fast development iteration",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed used with --sample (default: 42)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    _configure_logging(verbose=args.verbose)
    run_pipeline(
        args.input_json,
        args.output_json,
        sample=args.sample,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
