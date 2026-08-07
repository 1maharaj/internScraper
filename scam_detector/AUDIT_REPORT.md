# Audit Report: Scam Detector

<!--
=====================================================================
PHASE 1 — UNDERSTAND THE AIM
=====================================================================

This system functions as a fraud/scam detection layer for an AI-powered internship
aggregator. The aggregator normalizes scraped postings from Selenium scrapers into
a standardized JSON schema. The detector aims to safeguard users against fake posts,
advance-fee requests, credentials harvesting, and shell-company recruiters.

Schema Fields:
_id, name, company, applyLink, datePublished, deadlineDate, country, state, city,
isRemote, stipend, duration, skills, degree, field, experienceRequired, openings,
summary, responsibilities, perks, tags, source, isActive, createdAt, updatedAt.

Data-Quality Issues Handled:
1. `company` field mis-mapping: Occasional category/skill names leak into the field.
    identity-based checks (typosquatting, frequency) must verify `company_suspect` first.
2. `degree` field fallbacks: Scraper defaults to M.Tech/B.Tech for non-engineering roles.
    `degree_suspect_default` must be validated before using degree as a signal.
3. `deadlineDate` missingness: Almost all deadlines are null. Treat as unknown,
    never silently as zero urgency.

Target Architecture:
  load raw records
    -> data_quality.remediate_batch
    -> build corpus structures once (DuplicateIndex, peer groups, frequency tables)
    -> per-record feature extraction (text, company, url, stipend, temporal, structural)
    -> AnomalyModel.fit + .score (unsupervised IsolationForest)
    -> RulesEngine.run (noisy-OR combination)
    -> RiskEngine.score_record (blended score, decision, confidence, explanation)
    -> pipeline.run_pipeline (CLI entrypoint to score inputs)
    -> feedback.py (reviewer feedback loop store - not wired into pipeline yet)

Discrepancies identified:
- `extract_all` in `features/__init__.py` was incomplete (did not extract structural,
  stipend, or temporal feature slices, only text and company/url).
- `cross_company_duplicate_flag` in `duplicate_detection.py` compared company fields
  without checking if either was flagged as suspect.
- `TyposquatDomainRule` evaluated domain-company similarity even when the company name
  was suspect, causing false positive typosquat triggers.
-->

## Phase 2 — Audit Checklist

We executed `python -m pytest scam_detector/tests -v` on the existing test suite:
- **Status**: **PASS**
- **Summary**: 419 passed, 26 skipped (due to missing optional sentence-transformers package on baseline environment, all baseline tests pass).

### Checklist Status

*   **Package Structure & Config**:
    *   `scam_detector/config.py` exists with tunable thresholds/weights: **DONE**
    *   `scam_detector/data_quality/remediate.py` exists: **DONE**
*   **Data Quality (remediate.py)**:
    *   `flag_mislabeled_company`: **DONE**
    *   `flag_degree_default`: **DONE**
    *   `flag_missing_deadline`: **DONE**
    *   `clean_responsibilities`: **DONE**
    *   `flag_truncated_summary`: **DONE**
    *   `flag_inferred_date`: **DONE**
    *   `remediate_record` / `remediate_batch` with tests: **DONE**
*   **Text Features**:
    *   `urgency_score`: **DONE**
    *   `caps_and_punctuation_ratio`: **DONE**
    *   `genericity_score`: **DONE**
    *   `title_summary_alignment`: **DONE**
    *   `readability_and_grammar_signals`: **DONE**
    *   `sensitive_info_request_detector` (wired as HARD signal downstream): **DONE**
    *   `boilerplate_similarity`: **DONE**
    *   `extract_text_features` combining these: **DONE**
*   **Company & URL Features**:
    *   `is_company_suspect`, `has_legal_suffix`, `company_posting_frequency`, `typosquat_brand_distance`: **DONE**
    *   `parse_url_components`, `is_platform_internal_link`, `url_entropy`, `is_url_shortener`, `domain_company_name_similarity`, `is_known_ats_domain`: **DONE**
    *   `extract_company_url_features` combining both: **DONE**
*   **Stipend/Temporal/Structural Features**:
    *   `normalize_stipend_to_hourly_inr` (unpaid/performance handled separately): **DONE**
    *   `stipend_zscore` (peer-group-conditioned): **DONE**
    *   `stipend_perk_consistency_check`: **DONE**
    *   `posting_burst_score`: **DONE**
    *   `deadline_urgency_score` (returns None on missing, callers handle None): **DONE**
    *   `openings_zscore`, `field_completeness_score`: **DONE**
*   **Duplicate Detection**:
    *   `DuplicateIndex.build` / `find_near_duplicates` / `duplicate_cluster_report`: **DONE**
    *   `cross_company_duplicate_flag`: **PARTIAL** (gaps in checking `company_suspect` first)
    *   Embedding model name in `config.py`: **DONE**
*   **Rules Engine**:
    *   7 rules implemented (weights configurable, `UnverifiableCompanyRule` weighted LOW): **DONE**
    *   `TyposquatDomainRule` gate suspect company: **PARTIAL** (gaps in checking `company_is_suspect` first)
*   **Anomaly Model**:
    *   `AnomalyModel.fit` / `.score` / `.explain`: **DONE**
    *   `assemble_feature_matrix` helper: **DONE**
    *   `TextAnomalyModel` stub present: **DONE**
*   **Risk Engine**:
    *   `RiskEngine.score_record` produces `ScamScoreResult`: **DONE**
    *   Hard-disqualifying forces block/review (with test): **DONE**
    *   Low confidence forces review (with test): **DONE**
    *   `render_explanation` reviewer report: **DONE**
*   **Pipeline**:
    *   `run_pipeline` orchestration: **DONE**
    *   CLI entrypoint (`-m scam_detector.pipeline <input> <output>`): **DONE**
    *   `--sample N` flag: **DONE**
    *   Logging present: **DONE**
    *   Integration test in `test_pipeline.py` passes: **DONE**
*   **Feedback Loop**:
    *   `feedback.py` ReviewFeedback & FeedbackStore: **DONE**
    *   `build_training_labels` stub present: **DONE**
    *   Confirm NOT wired into main pipeline: **DONE**
*   **Cross-cutting checks**:
    *   Check `company_suspect` first: **PARTIAL** (needs updates to `cross_company_duplicate_flag` and `TyposquatDomainRule`)
    *   Check `degree_suspect_default` first: **DONE**
    *   Null deadline as unknown (coerces to None, never 0): **DONE**
    *   No silent data mutation / overwrite: **DONE**
    *   All weights/thresholds in `config.py`: **PARTIAL** (`pipeline.py` hardcodes `is_outlier_high`/`is_outlier_low` z-score thresholds; `extract_all` feature extractor is incomplete).
