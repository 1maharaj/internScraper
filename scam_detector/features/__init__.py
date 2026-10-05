"""
scam_detector.features
======================
Feature extraction layer.  Each module is responsible for one domain of
signals and populates its slice of the shared :class:`FeatureVector`.

Exports
-------
  FeatureVector              — Aggregated feature vector (all domains)
  TextFeatureVector          — Detailed text-feature model (standalone use)
  CompanyUrlFeatureVector    — Combined company + URL feature vector
  extract_all                — Convenience: run every enabled extractor
  extract_text_features      — Text extractor
  extract_company_url_features — Company + URL extractor
"""

from scam_detector.features.text_features import (
    TextFeatures,
    TextFeatureVector,
    extract_text_features,
    urgency_score,
    caps_and_punctuation_ratio,
    genericity_score,
    title_summary_alignment,
    readability_and_grammar_signals,
    sensitive_info_request_detector,
    guaranteed_outcome_claim_detector,
    external_form_and_phone_handoff_detector,
    training_program_disguised_as_internship_detector,
    zero_shot_scam_signal,
    boilerplate_similarity,
)
from scam_detector.features.company_features import (
    CompanyFeatures,
    CompanyUrlFeatureVector,
    extract_company_url_features,
    is_company_suspect,
    has_legal_suffix,
    company_posting_frequency,
    typosquat_brand_distance,
)
from scam_detector.features.url_features import (
    UrlFeatures,
    parse_url_components,
    is_platform_internal_link,
    url_entropy,
    is_url_shortener,
    domain_company_name_similarity,
    is_known_ats_domain,
)
from scam_detector.features.stipend_features import StipendFeatures
from scam_detector.features.temporal_features import TemporalFeatures
from scam_detector.features.structural_features import StructuralFeatures
from scam_detector.features.duplicate_detection import (
    DuplicateIndex,
    DuplicateMatch,
    ClusterReport,
    cross_company_duplicate_flag,
)
from scam_detector.features.graph_features import (
    build_company_infrastructure_graph,
    shared_infrastructure_flag,
    duplicate_cluster_network_size,
    export_largest_components_visualization,
    compute_graph_network_metrics,
    company_network_risk_profile,
)

from pydantic import BaseModel, Field


class FeatureVector(BaseModel):
    """
    Aggregated feature vector produced by all extractors.
    Passed directly into the scoring layer.
    """

    text: TextFeatureVector = Field(default_factory=TextFeatureVector)
    company: CompanyFeatures = Field(default_factory=CompanyFeatures)
    url: UrlFeatures = Field(default_factory=UrlFeatures)
    stipend: StipendFeatures = Field(default_factory=StipendFeatures)
    temporal: TemporalFeatures = Field(default_factory=TemporalFeatures)
    structural: StructuralFeatures = Field(default_factory=StructuralFeatures)


def extract_all(
    record: object,
    all_records: list | None = None,
) -> FeatureVector:
    """
    Run all feature extractors against a remediated record and return a
    unified :class:`FeatureVector`.
    """
    from typing import Any
    from scam_detector.config import cfg
    from scam_detector.features.stipend_features import (
        normalize_stipend_to_hourly_inr,
        stipend_zscore,
        stipend_perk_consistency_check,
    )
    from scam_detector.features.temporal_features import (
        posting_burst_score,
        deadline_urgency_score,
    )
    from scam_detector.features.structural_features import (
        field_completeness_score,
        openings_zscore,
    )

    raw: dict = record.record if hasattr(record, "record") else record  # type: ignore[union-attr]
    flags: dict = record.flags if hasattr(record, "flags") else {}      # type: ignore[union-attr]

    raw_with_flags = {**raw, "_flags": flags}
    batch = all_records or [raw]

    # Company / URL
    cu = extract_company_url_features(raw_with_flags, all_records=batch, flags=flags)

    # Stipend
    hourly = normalize_stipend_to_hourly_inr(raw.get("stipend") or {}, raw.get("duration") or {})
    
    # Peer group logic for z-scores
    from scam_detector.features.stipend_features import (
        get_role_category,
        get_city_tier,
        get_company_size_tier,
    )

    def _build_peer_group(rec: dict[str, Any], corpus: list[dict[str, Any]]) -> list[dict[str, Any]]:
        target_role = get_role_category(rec)
        target_city = get_city_tier(rec)
        target_size = get_company_size_tier(rec)

        peers = [
            r for r in corpus
            if get_role_category(r) == target_role
            and get_city_tier(r) == target_city
            and get_company_size_tier(r) == target_size
        ]
        if len(peers) < 2:
            peers = [
                r for r in corpus
                if get_role_category(r) == target_role
                and get_city_tier(r) == target_city
            ]
        if len(peers) < 2:
            peers = [
                r for r in corpus
                if get_role_category(r) == target_role
            ]
        return peers if peers else list(corpus)

    peer_group = _build_peer_group(raw, batch)
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
        is_outlier_high=bool(peer_z is not None and peer_z > cfg.rule_thresholds.stipend_zscore_threshold),
        is_outlier_low=bool(peer_z is not None and peer_z < -cfg.rule_thresholds.stipend_zscore_threshold),
        amount_plausibility_score=1.0 if hourly is not None else 0.5,
        missing_stipend_for_role=hourly is None,
    )

    # Temporal
    company_key = (raw.get("company") or "").strip().lower()
    company_recs = [
        r for r in batch
        if (r.get("company") or "").strip().lower() == company_key
    ] if company_key else [raw]
    
    burst = posting_burst_score(raw, company_recs)
    from scam_detector.features.temporal_features import recruiter_posting_velocity
    v24 = recruiter_posting_velocity(raw, batch, hours=24)
    v72 = recruiter_posting_velocity(raw, batch, hours=72)
    temporal = TemporalFeatures(
        posting_burst_count=int(burst.get("burst_count") or 0),
        posting_burst_cadence=burst.get("cadence_days"),
        deadline_urgency_score=deadline_urgency_score(raw),
        recruiter_posting_velocity_24h=v24,
        recruiter_posting_velocity_72h=v72,
    )


    # Structural
    completeness = field_completeness_score(raw)
    oz = openings_zscore(raw, peer_group)
    skills = raw.get("skills") or []
    skills_count = len(skills) if isinstance(skills, list) else 0
    resp = raw.get("responsibilities") or []
    resp_count = len(resp) if isinstance(resp, list) else 0

    structural = StructuralFeatures(
        openings_zscore=oz,
        field_completeness=completeness,
        completeness_ratio=completeness,
        skills_count=skills_count,
        skills_count_anomaly=skills_count < 1 or skills_count > 30,
        responsibilities_count=resp_count,
    )

    return FeatureVector(
        text=extract_text_features(raw_with_flags),
        company=cu.company,
        url=cu.url,
        stipend=stipend,
        temporal=temporal,
        structural=structural,
    )


__all__: list[str] = [
    "FeatureVector",
    "TextFeatureVector",
    "TextFeatures",
    "CompanyFeatures",
    "UrlFeatures",
    "CompanyUrlFeatureVector",
    "StipendFeatures",
    "TemporalFeatures",
    "StructuralFeatures",
    "extract_all",
    "extract_text_features",
    "extract_company_url_features",
    "is_company_suspect",
    "has_legal_suffix",
    "company_posting_frequency",
    "typosquat_brand_distance",
    "parse_url_components",
    "is_platform_internal_link",
    "url_entropy",
    "is_url_shortener",
    "domain_company_name_similarity",
    "is_known_ats_domain",
    "DuplicateIndex",
    "DuplicateMatch",
    "ClusterReport",
    "cross_company_duplicate_flag",
    "build_company_infrastructure_graph",
    "shared_infrastructure_flag",
    "duplicate_cluster_network_size",
    "export_largest_components_visualization",
    "compute_graph_network_metrics",
    "company_network_risk_profile",
    "urgency_score",
    "caps_and_punctuation_ratio",
    "genericity_score",
    "title_summary_alignment",
    "readability_and_grammar_signals",
    "sensitive_info_request_detector",
    "guaranteed_outcome_claim_detector",
    "external_form_and_phone_handoff_detector",
    "training_program_disguised_as_internship_detector",
    "zero_shot_scam_signal",
    "boilerplate_similarity",
]
