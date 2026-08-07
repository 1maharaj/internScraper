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
from datetime import datetime
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
    min_jaccard: float = _PEER_JACCARD_MIN,
) -> list[dict[str, Any]]:
    """
    Peer group for stipend / openings z-scores: same ``isRemote`` plus
    field/tags Jaccard overlap.  Includes *record* itself when present in
    ``all_records`` so σ estimates use the full category cohort.
    """
    own_labels = _as_str_set(record.get("field")) | _as_str_set(record.get("tags"))
    own_remote = bool(record.get("isRemote"))
    peers: list[dict[str, Any]] = []
    for other in all_records:
        if bool(other.get("isRemote")) != own_remote:
            continue
        other_labels = _as_str_set(other.get("field")) | _as_str_set(other.get("tags"))
        if not own_labels or not other_labels:
            # Fall back to same-remote cohort when labels are missing so tiny
            # fixtures still get a usable peer set.
            peers.append(other)
            continue
        if _jaccard(own_labels, other_labels) >= min_jaccard:
            peers.append(other)
    return peers if peers else list(all_records)


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
) -> FeatureVector:
    """Per-record feature extraction using pre-built corpus structures."""
    raw = remediated.record
    flags = remediated.flags
    raw_with_flags = {**raw, "_flags": flags}

    text = extract_text_features(raw_with_flags)
    cu = extract_company_url_features(
        raw_with_flags, all_records=all_records, flags=flags
    )

    hourly = normalize_stipend_to_hourly_inr(
        raw.get("stipend") or {},
        raw.get("duration") or {},
    )
    peer_z = stipend_zscore(raw, peer_group)
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
        is_outlier_high=bool(peer_z is not None and peer_z > 3.0),
        is_outlier_low=bool(peer_z is not None and peer_z < -2.0),
        amount_plausibility_score=1.0 if hourly is not None else 0.5,
        missing_stipend_for_role=hourly is None,
    )

    burst = posting_burst_score(raw, company_records)
    temporal = TemporalFeatures(
        posting_burst_count=int(burst.get("burst_count") or 0),
        posting_burst_cadence=burst.get("cadence_days"),
        deadline_urgency_score=deadline_urgency_score(raw),
    )

    completeness = field_completeness_score(raw)
    oz = openings_zscore(raw, peer_group)
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
    flags: dict[str, Any] | None = None,
) -> RuleInput:
    """Explicit bridge from FeatureVector → RuleInput (avoids duck-type gaps)."""
    return RuleInput(
        sensitive_info_requested=fv.text.sensitive_info_requested,
        urgency_score=fv.text.urgency_score,
        genericity_score=fv.text.genericity_score,
        caps_ratio=fv.text.caps_ratio,
        exclamation_count=fv.text.exclamation_count,
        has_repeated_punctuation=fv.text.has_repeated_punctuation,
        summary_truncated=fv.text.summary_truncated,
        company_is_suspect=fv.company.is_suspect,
        typosquat_min_distance=fv.company.typosquat_min_distance,
        is_platform_internal=fv.url.is_platform_internal,
        is_url_shortener=fv.url.is_url_shortener,
        is_known_ats=fv.url.is_known_ats,
        domain_company_similarity=fv.url.domain_company_similarity,
        stipend_peer_zscore=fv.stipend.peer_zscore,
        perk_consistency_ok=fv.stipend.perk_consistency_ok,
        openings_zscore=fv.structural.openings_zscore,
        field_completeness=fv.structural.field_completeness,
        cross_company_duplicate=cross_company_duplicate,
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
        records[i] = rec
        remediated[i] = RemediatedRecord(record=rec, flags=flags_list[i])

    # ── Corpus structures (once) ──────────────────────────────────────────
    neighbors_by_id = _build_duplicate_neighbors(records)
    by_company = _index_by_company(records)
    peer_cache: list[list[dict[str, Any]]] = [
        build_peer_group(rec, records) for rec in records
    ]

    # ── Per-record features (Prompts 2–5) ─────────────────────────────────
    feature_vectors: list[FeatureVector] = []
    cross_company_flags: list[bool] = []

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
        )
        feature_vectors.append(fv)

        dup_neighbors = neighbors_by_id.get(rid, [])
        cross_company_flags.append(
            cross_company_duplicate_flag(rec, dup_neighbors, records)
        )

    # ── Prompt 7: anomaly model ───────────────────────────────────────────
    matrix = assemble_feature_matrix(records, feature_vectors)
    anomaly_scores = [0.0] * n
    anomaly_explanations: list[list[tuple[str, float]]] = [[] for _ in range(n)]

    if n >= 2 and not matrix.empty:
        try:
            model = AnomalyModel(random_state=42, contamination="auto")
            model.fit(matrix)
            scored = model.score(matrix)
            anomaly_scores = [float(s) for s in scored]
            for i in range(n):
                anomaly_explanations[i] = model.explain(matrix.iloc[i])
            log.info("AnomalyModel fitted on %d rows", n)
        except Exception as exc:
            log.warning("AnomalyModel failed (%s) — anomaly scores default to 0.0", exc)
    else:
        log.info("Skipping AnomalyModel (need ≥2 records); anomaly scores = 0.0")

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
            flags=rem.flags,
        )
        rule_result = rules_engine.run(rule_input)
        confidence = compute_confidence_score(rem, fv)
        result: ScamScoreResult = risk_engine.score_record(
            record=rem,
            rule_result=rule_result,
            anomaly_score=anomaly_scores[i],
            confidence_score=confidence,
            feature_contributions=anomaly_explanations[i],
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
            "reviewedAt": datetime.utcnow().isoformat(),
            "rejectionReason": result.explanation_summary if result.decision == "block" else None,
        }
        outputs.append(out)
        decisions.append(result.decision)

    bucket_counts = Counter(decisions)
    log.info(
        "Decision buckets — clear=%d review=%d block=%d",
        bucket_counts.get("clear", 0),
        bucket_counts.get("review", 0),
        bucket_counts.get("block", 0),
    )
    return outputs


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
