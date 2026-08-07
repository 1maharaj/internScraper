"""
test_pipeline.py
----------------
End-to-end integration test for the Phase 6 batch pipeline.

Uses a small synthetic fixture (not internships.json) so CI stays fast.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scam_detector.config import Config
from scam_detector.pipeline import (
    process_records,
    run_pipeline,
    load_records,
    feature_vector_to_rule_input,
)
from scam_detector.features import FeatureVector
from scam_detector import ScamDetectorPipeline


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

_DUP_SUMMARY = (
    "Join our grassroots fundraising internship supporting education and "
    "community development. You will research donors, draft outreach emails, "
    "assist with campaign planning, maintain donor databases, and collaborate "
    "with the core team on stakeholder engagement and digital promotion. "
    "Performance-based stipend with weekly targets."
)


def _base(
    *,
    _id: str,
    name: str,
    company: str,
    summary: str,
    stipend_amount: int = 10000,
    openings: int = 2,
    apply: str | None = None,
    sparse: bool = False,
    extra: dict | None = None,
) -> dict:
    if sparse:
        return {
            "_id": _id,
            "name": name,
            "company": company,
            "summary": summary,
            "applyLink": apply or "https://bit.ly/xyz",
        }

    rec = {
        "_id": _id,
        "name": name,
        "company": company,
        "summary": summary,
        "applyLink": apply or f"https://internshala.com/internship/detail/{_id}",
        "datePublished": "2026-05-01",
        "deadlineDate": "2026-06-15",
        "country": "India",
        "state": "Maharashtra",
        "city": "Mumbai",
        "isRemote": True,
        "stipend": {
            "type": "paid",
            "amount": {"min": stipend_amount, "max": stipend_amount, "period": "month"},
            "currency": "INR",
        },
        "duration": {"value": 3, "unit": "months"},
        "skills": ["Python", "Communication", "Excel"],
        "degree": ["B.Tech", "B.E"],
        "field": ["Computer Science", "Software Development"],
        "tags": ["internship", "remote", "software"],
        "experienceRequired": "0-1 years",
        "openings": openings,
        "responsibilities": [
            "Build features",
            "Write tests",
            "Collaborate with mentors",
        ],
        "perks": ["Certificate", "Letter of recommendation"],
        "source": "test_fixture",
        "isActive": True,
    }
    if extra:
        rec.update(extra)
    return rec


def pipeline_fixture_records() -> list[dict]:
    """
    ~16 synthetic records covering:
      - clean high-confidence listing
      - hard-disqualifying (payment / Aadhaar request)
      - cross-company near-duplicate pair
      - low-confidence / sparse listing
      - peer cohort fillers for z-scores / anomaly fit
    """
    clean_summary = (
        "Work with our engineering team on backend APIs in Python. "
        "You will ship small features, review pull requests, and learn "
        "production debugging practices under mentorship."
    )

    records = [
        # 0 — CLEAN (expected: clear)
        _base(
            _id="clean-001",
            name="Backend Development Intern",
            company="BrightLabs Pvt Ltd",
            summary=clean_summary,
            stipend_amount=12000,
            openings=2,
        ),
        # 1 — HARD DISQUALIFYING (expected: block)
        _base(
            _id="hard-dq-001",
            name="Marketing Intern",
            company="QuickHire Shell",
            summary=(
                "Exciting opportunity! Candidates must pay a security deposit "
                "of Rs 2000 and submit Aadhaar card copy before onboarding. "
                "URGENT hiring — apply now!!!"
            ),
            stipend_amount=25000,
            openings=50,
            apply="https://totally-legit-jobs.xyz/apply",
        ),
        # 2 & 3 — CROSS-COMPANY DUPLICATE PAIR (expected: review or block)
        _base(
            _id="dup-shell-a",
            name="Fundraising Intern",
            company="Shell Org Alpha",
            summary=_DUP_SUMMARY,
            stipend_amount=5000,
            openings=20,
            extra={"field": ["Non-Profit", "Fundraising"], "tags": ["internship", "remote"]},
        ),
        _base(
            _id="dup-shell-b",
            name="Fundraising Intern",
            company="Shell Org Beta",
            summary=_DUP_SUMMARY,
            stipend_amount=5000,
            openings=20,
            extra={"field": ["Non-Profit", "Fundraising"], "tags": ["internship", "remote"]},
        ),
        # 4 — SPARSE / LOW CONFIDENCE (expected: review, never clear)
        _base(
            _id="sparse-001",
            name="Intern",
            company="Digital Marketing",  # category-leak style
            summary="Internship available.",
            sparse=True,
        ),
    ]

    # Peer fillers — normal SWE-like postings so clean-001 has a cohort
    for i in range(5, 16):
        records.append(
            _base(
                _id=f"peer-{i:02d}",
                name="Software Development Intern",
                company=f"PeerCo {i} Technologies",
                summary=(
                    f"Peer listing {i}: assist with software development tasks, "
                    "write documentation, and participate in code reviews with "
                    "the engineering team on real product work."
                ),
                stipend_amount=8000 + (i * 200),
                openings=1 + (i % 3),
            )
        )

    return records


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestProcessRecordsIntegration:
    def test_expected_decision_buckets(self) -> None:
        records = pipeline_fixture_records()
        assert 10 <= len(records) <= 20

        outputs = process_records(records)
        assert len(outputs) == len(records)

        by_id = {o["_id"]: o for o in outputs}

        # Order preserved
        assert [o["_id"] for o in outputs] == [r["_id"] for r in records]

        # Required output fields
        for out in outputs:
            assert "scam_score" in out
            assert "decision" in out
            assert "explanation_summary" in out
            assert "confidence" in out
            assert out["decision"] in ("clear", "review", "block")
            assert 0.0 <= out["scam_score"] <= 100.0
            assert 0.0 <= out["confidence"] <= 1.0

        clean = by_id["clean-001"]
        hard = by_id["hard-dq-001"]
        dup_a = by_id["dup-shell-a"]
        dup_b = by_id["dup-shell-b"]
        sparse = by_id["sparse-001"]

        assert clean["decision"] == "clear", (
            f"clean record should be clear, got {clean['decision']} "
            f"(score={clean['scam_score']}, conf={clean['confidence']}, "
            f"summary={clean['explanation_summary']!r})"
        )

        assert hard["decision"] in ("block", "review"), (
            f"hard-DQ must be block/review, got {hard['decision']}"
        )
        assert hard["decision"] == "block"

        assert dup_a["decision"] in ("review", "block"), (
            f"duplicate A expected review/block, got {dup_a['decision']} "
            f"({dup_a['explanation_summary']!r})"
        )
        assert dup_b["decision"] in ("review", "block"), (
            f"duplicate B expected review/block, got {dup_b['decision']} "
            f"({dup_b['explanation_summary']!r})"
        )

        assert sparse["decision"] != "clear", (
            f"sparse/low-confidence must not be clear, got {sparse['decision']}"
        )
        assert sparse["decision"] in ("review", "block")
        assert sparse["confidence"] < 0.4


class TestRunPipelineIO:
    def test_run_pipeline_writes_json(self, tmp_path: Path) -> None:
        records = pipeline_fixture_records()
        inp = tmp_path / "in.json"
        out = tmp_path / "out.json"
        inp.write_text(json.dumps(records), encoding="utf-8")

        run_pipeline(str(inp), str(out), sample=None)
        loaded = load_records(out)
        assert len(loaded) == len(records)
        assert all("decision" in r for r in loaded)

    def test_sample_flag_reduces_count(self, tmp_path: Path) -> None:
        records = pipeline_fixture_records()
        inp = tmp_path / "in.json"
        out = tmp_path / "out.json"
        inp.write_text(json.dumps({"internships": records}), encoding="utf-8")

        run_pipeline(str(inp), str(out), sample=5, seed=0)
        loaded = load_records(out)
        assert len(loaded) == 5


class TestSingleRecordCompat:
    def test_scam_detector_pipeline_still_works(self) -> None:
        raw = pipeline_fixture_records()[0]
        result = ScamDetectorPipeline().run(raw)
        assert result.decision in ("clear", "review", "block")
        assert 0.0 <= result.scam_score <= 100.0

    def test_feature_vector_to_rule_input_defaults(self) -> None:
        inp = feature_vector_to_rule_input(
            FeatureVector(),
            cross_company_duplicate=False,
        )
        assert inp.cross_company_duplicate is False
        assert inp.perk_consistency_ok is True
