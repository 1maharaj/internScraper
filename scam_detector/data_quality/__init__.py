"""
scam_detector.data_quality
==========================
Phase 1 of the pipeline: clean and normalise raw internship data
before any feature extraction takes place.

Exports
-------
  RemediatedRecord         — Pydantic model: cleaned record + flags dict
  remediate_record         — Remediate a single internship dict
  remediate_batch          — Remediate a list of internship dicts
  flag_mislabeled_company  — Fix 1: detect category/skill string in company field
  flag_degree_default      — Fix 2: detect scraper-fallback degree combos
  flag_missing_deadline    — Fix 3: flag null deadlineDate
  clean_responsibilities   — Fix 4: strip stray numbered-list artefacts
  flag_truncated_summary   — Fix 5: detect hard-truncated summaries
  flag_inferred_date       — Fix 6: detect datetime.now() fallback dates
"""

from scam_detector.data_quality.remediate import (
    RemediatedRecord,
    remediate_record,
    remediate_batch,
    flag_mislabeled_company,
    flag_degree_default,
    flag_missing_deadline,
    clean_responsibilities,
    flag_truncated_summary,
    flag_inferred_date,
)

# Legacy alias kept for the smoke-tests written against the old skeleton
remediate = remediate_record

__all__: list[str] = [
    "RemediatedRecord",
    "remediate_record",
    "remediate_batch",
    "remediate",          # alias
    "flag_mislabeled_company",
    "flag_degree_default",
    "flag_missing_deadline",
    "clean_responsibilities",
    "flag_truncated_summary",
    "flag_inferred_date",
]
