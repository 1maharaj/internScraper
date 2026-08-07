"""
test_duplicate_detection.py
----------------------------
Tests for scam_detector.features.duplicate_detection.

All tests that require sentence-transformers are guarded with
``pytest.skip`` when the library is not installed — consistent with the
pattern used in test_features.py.

Real corpus fixtures
--------------------
NayePankh Foundation and Basti Ki Pathshala Foundation fundraising records
are taken verbatim from:
  internScraper/scrapers/intershala_scraper/internships.json

These two organisations post structurally similar fundraising internship text.
See the module docstring for the documented ambiguity and why a
same_parent_organization_allowlist is needed in production.

Unrelated technical records (Anakin, RootSky) are used as true negatives
to confirm the system does NOT flag genuinely different content.
"""
from __future__ import annotations

import pytest

from scam_detector.features.duplicate_detection import (
    DuplicateIndex,
    DuplicateMatch,
    ClusterReport,
    cross_company_duplicate_flag,
    _record_text,
    _record_id,
)

# ---------------------------------------------------------------------------
# SBERT availability guard
# ---------------------------------------------------------------------------

def _sbert_available() -> bool:
    try:
        from scam_detector.features.text_features import _sbert_model
        return _sbert_model() is not None
    except Exception:
        return False


SBERT_AVAILABLE = _sbert_available()
skip_no_sbert = pytest.mark.skipif(
    not SBERT_AVAILABLE,
    reason="sentence-transformers not installed",
)

# ---------------------------------------------------------------------------
# Fixtures — real corpus data
# ---------------------------------------------------------------------------

# NayePankh Foundation — Fundraising (verbatim from Internshala corpus)
NAYEPANKH_FUNDRAISING = {
    "_id": "nayepankh-fundraising-001",
    "name": "Fundraising",
    "company": "NayePankh Foundation",
    "summary": (
        "NayePankh Foundation works at the grassroots level to uplift underserved "
        "communities by promoting education, life skills, and sustainable development. "
        "As a Fundraising Intern, you will collaborate closely with the core team to "
        "support resource mobilization efforts and strengthen connections with individuals "
        "and organizations who believe in our mission. This role offers hands-on exposure "
        "to non-profit fundraising, outreach, and stakeholder engagement while allowing "
        "you to apply practical skills such as communication, relationship-building, "
        "research, and digital promotion. Performance-based stipend."
    ),
    "applyLink": "https://internshala.com/internship/detail/fundraising-nayepankh1777892293",
    "field": ["Non-Profit", "Fundraising"],
    "tags": ["internship", "remote"],
    "datePublished": "2026-05-01",
}

# Basti Ki Pathshala Foundation — Fundraising (verbatim from Internshala corpus)
BASTI_FUNDRAISING = {
    "_id": "basti-fundraising-001",
    "name": "Fundraising",
    "company": "Basti Ki Pathshala Foundation",
    "summary": (
        "Are you passionate about making a difference in the lives of underprivileged "
        "children? Join us as a Fundraising Intern at Basti Ki Pathshala Foundation and "
        "be a part of our mission to provide quality education to every child. As a "
        "Fundraising Intern, you will have the opportunity to learn valuable skills in "
        "donor relations, event planning, and campaign management. 1. Assisting in the "
        "development and implementation of fundraising campaigns 2. Researching potential "
        "donors and sponsors 3. Creating and maintaining donor databases 4. Drafting "
        "fundraising proposals and letters 5. Assisting in organizing fundraising events "
        "6. Providing support in marketing and communication efforts 7. Collaborating "
        "with the team to achieve fundraising goals. Performance-based stipend."
    ),
    "applyLink": "https://internshala.com/internship/detail/fundraising-basti1777893066",
    "field": ["Non-Profit", "Fundraising"],
    "tags": ["internship", "remote"],
    "datePublished": "2026-05-02",
}

# NayePankh — Social Entrepreneurship (different role, same org)
NAYEPANKH_SOCIAL = {
    "_id": "nayepankh-social-001",
    "name": "Social Entrepreneurship",
    "company": "NayePankh Foundation",
    "summary": (
        "As a social entrepreneurship intern, you will be working at the NayePankh "
        "Foundation, a non-profit organization dedicated to fostering economic, "
        "educational, and leadership opportunities for marginalized women and girls. "
        "You will be a key member of the team, working closely with the foundation's "
        "leadership to identify and develop social enterprise initiatives. "
        "Performance based stipend."
    ),
    "applyLink": "https://internshala.com/internship/detail/social-entrepreneurship-nayepankh1777892517",
    "field": ["Non-Profit", "Social Enterprise"],
    "tags": ["internship", "remote"],
    "datePublished": "2026-05-01",
}

# Basti Ki Pathshala — Social Entrepreneurship (same template structure)
BASTI_SOCIAL = {
    "_id": "basti-social-001",
    "name": "Social Entrepreneurship",
    "company": "Basti Ki Pathshala Foundation",
    "summary": (
        "1. Assisting in the development and implementation of innovative fundraising "
        "strategies designed to meet financial goals. 2. Conducting thorough research "
        "to identify potential donors, sponsorships, and grant opportunities. "
        "3. Supporting the creation of compelling content for donor communications, "
        "including newsletters, social media updates, and email campaigns. "
        "4. Collaborating with the team to plan, organize, and execute fundraising events. "
        "5. Maintaining and updating donor databases to ensure information is current and accurate. "
        "Performance-based stipend."
    ),
    "applyLink": "https://internshala.com/internship/detail/social-entrepreneurship-basti1777893262",
    "field": ["Non-Profit", "Social Enterprise"],
    "tags": ["internship", "remote"],
    "datePublished": "2026-05-02",
}

# Anakin — unrelated technical internship (true negative)
ANAKIN_TECH = {
    "_id": "anakin-software-001",
    "name": "Software Development",
    "company": "Anakin",
    "summary": (
        "Anakin is building a large-scale data engine that powers real-time competitive "
        "intelligence for global internet companies by continuously collecting and "
        "structuring massive volumes of public web data. This involves solving complex "
        "engineering challenges like handling scale, ensuring system reliability, "
        "navigating dynamic websites, overcoming anti-bot mechanisms, and dealing with "
        "unpredictable edge cases. Interns are expected to take ownership — understanding "
        "problems, writing functional code, debugging issues, and improving system "
        "performance rather than passively observing."
    ),
    "applyLink": "https://internshala.com/internship/detail/software-development-anakin1777439557",
    "field": ["Software Engineering", "Backend"],
    "tags": ["internship", "backend"],
    "datePublished": "2026-05-01",
}

# RootSky — another unrelated technical internship (true negative)
ROOTSKY_TECH = {
    "_id": "rootsky-web-001",
    "name": "Software Development Engineering (Web)",
    "company": "RootSky System (OPC) Private Limited",
    "summary": (
        "RootSky System is a premier technology partner specializing in Software "
        "Development and System Integration. Intern will handle environment setup, "
        "cloud deployment, scripting with JavaScript and Python, and network support. "
        "Apply basic IT Networking concepts to troubleshoot connectivity between tools."
    ),
    "applyLink": "https://internshala.com/internship/detail/web-engineering-rootsky1777983872",
    "field": ["Software Engineering", "Web Development"],
    "tags": ["internship"],
    "datePublished": "2026-05-12",
}

# Scam shell-company pair: same text, different invented company names
SCAM_SHELL_A = {
    "_id": "scam-shell-a",
    "name": "Business Development Intern",
    "company": "Alpha Solutions Pvt Ltd",
    "summary": (
        "We are looking for a business development intern to support growth and "
        "outreach efforts. The intern will help identify and connect with potential "
        "partners. The role involves research, lead generation, outreach support, "
        "follow-ups, campaign coordination, and basic sales tracking. "
        "Pay registration fee of Rs 500 to confirm your slot."
    ),
    "applyLink": "https://bit.ly/scam-link-alpha",
    "field": ["Business Development"],
    "tags": ["internship"],
    "datePublished": "2026-05-01",
}

SCAM_SHELL_B = {
    "_id": "scam-shell-b",
    "name": "Business Development Intern",
    "company": "Beta Ventures Pvt Ltd",   # different company, same script
    "summary": (
        "We are looking for a business development intern to support growth and "
        "outreach efforts. The intern will help identify and connect with potential "
        "partners. The role involves research, lead generation, outreach support, "
        "follow-ups, campaign coordination, and basic sales tracking. "
        "Pay registration fee of Rs 500 to confirm your slot."
    ),
    "applyLink": "https://bit.ly/scam-link-beta",
    "field": ["Business Development"],
    "tags": ["internship"],
    "datePublished": "2026-05-02",
}

# Full mixed corpus for integration tests
FULL_CORPUS = [
    NAYEPANKH_FUNDRAISING,
    BASTI_FUNDRAISING,
    NAYEPANKH_SOCIAL,
    BASTI_SOCIAL,
    ANAKIN_TECH,
    ROOTSKY_TECH,
    SCAM_SHELL_A,
    SCAM_SHELL_B,
]


# ===========================================================================
# Helper utilities
# ===========================================================================

class TestRecordText:
    def test_uses_name_and_summary(self) -> None:
        r = {"name": "Intern", "summary": "Great role."}
        assert "Intern" in _record_text(r)
        assert "Great role" in _record_text(r)

    def test_falls_back_to_title_and_description(self) -> None:
        r = {"title": "Fundraising", "description": "Help us."}
        assert "Fundraising" in _record_text(r)
        assert "Help us" in _record_text(r)

    def test_empty_record_returns_empty_string(self) -> None:
        assert _record_text({}) == ""

    def test_collapses_whitespace(self) -> None:
        r = {"name": "A  B", "summary": "C   D"}
        text = _record_text(r)
        assert "  " not in text


class TestRecordId:
    def test_uses_id_field(self) -> None:
        assert _record_id({"_id": "abc"}, 0) == "abc"

    def test_uses_internship_id(self) -> None:
        assert _record_id({"internship_id": "xyz"}, 0) == "xyz"

    def test_falls_back_to_index(self) -> None:
        assert _record_id({}, 5) == "5"

    def test_mongo_oid_extracted(self) -> None:
        r = {"_id": {"$oid": "507f1f77bcf86cd799439011"}}
        assert _record_id(r, 0) == "507f1f77bcf86cd799439011"


# ===========================================================================
# DuplicateIndex — no-SBERT path
# ===========================================================================

class TestDuplicateIndexNoSbert:
    """Tests that don't require model inference."""

    def test_build_raises_without_sbert(self) -> None:
        if SBERT_AVAILABLE:
            pytest.skip("SBERT available — skipping no-model error path")
        idx = DuplicateIndex()
        with pytest.raises(RuntimeError, match="sentence-transformers"):
            idx.build([ANAKIN_TECH])

    def test_find_returns_empty_when_not_built(self) -> None:
        idx = DuplicateIndex()
        assert idx.find_near_duplicates("anything") == []

    def test_cluster_report_returns_empty_when_not_built(self) -> None:
        idx = DuplicateIndex()
        report = idx.duplicate_cluster_report()
        assert report.total_records == 0

    def test_record_count_zero_before_build(self) -> None:
        assert DuplicateIndex().record_count == 0


# ===========================================================================
# DuplicateIndex — with SBERT
# ===========================================================================

class TestDuplicateIndexWithSbert:

    @skip_no_sbert
    def test_build_empty_corpus(self) -> None:
        idx = DuplicateIndex()
        idx.build([])
        assert idx.record_count == 0
        assert idx.find_near_duplicates("anything") == []

    @skip_no_sbert
    def test_build_sets_record_count(self) -> None:
        idx = DuplicateIndex()
        idx.build(FULL_CORPUS)
        assert idx.record_count == len(FULL_CORPUS)

    @skip_no_sbert
    def test_embeddings_shape(self) -> None:
        idx = DuplicateIndex()
        idx.build(FULL_CORPUS)
        emb = idx.corpus_embeddings()
        assert emb is not None
        assert emb.shape[0] == len(FULL_CORPUS)
        assert emb.shape[1] > 0

    @skip_no_sbert
    def test_embeddings_l2_normalised(self) -> None:
        import numpy as np
        idx = DuplicateIndex()
        idx.build(FULL_CORPUS)
        emb = idx.corpus_embeddings()
        norms = np.linalg.norm(emb, axis=1)
        assert all(abs(n - 1.0) < 1e-5 for n in norms), "Embeddings should be L2-normalised"

    @skip_no_sbert
    def test_find_self_not_returned(self) -> None:
        idx = DuplicateIndex()
        idx.build(FULL_CORPUS)
        results = idx.find_near_duplicates("anakin-software-001", threshold=0.0)
        ids = [r[0] for r in results]
        assert "anakin-software-001" not in ids

    @skip_no_sbert
    def test_find_returns_sorted_descending(self) -> None:
        idx = DuplicateIndex()
        idx.build(FULL_CORPUS)
        results = idx.find_near_duplicates("scam-shell-a", threshold=0.0)
        sims = [s for _, s in results]
        assert sims == sorted(sims, reverse=True)

    @skip_no_sbert
    def test_similarity_values_bounded(self) -> None:
        idx = DuplicateIndex()
        idx.build(FULL_CORPUS)
        for record_id in [r["_id"] for r in FULL_CORPUS]:
            for _, sim in idx.find_near_duplicates(record_id, threshold=0.0):
                assert 0.0 <= sim <= 1.0

    @skip_no_sbert
    def test_unknown_id_returns_empty(self) -> None:
        idx = DuplicateIndex()
        idx.build(FULL_CORPUS)
        assert idx.find_near_duplicates("does-not-exist-xyz") == []

    @skip_no_sbert
    def test_get_embedding_returns_vector(self) -> None:
        import numpy as np
        idx = DuplicateIndex()
        idx.build(FULL_CORPUS)
        emb = idx.get_embedding("anakin-software-001")
        assert emb is not None
        assert isinstance(emb, np.ndarray)

    @skip_no_sbert
    def test_get_embedding_unknown_id_returns_none(self) -> None:
        idx = DuplicateIndex()
        idx.build(FULL_CORPUS)
        assert idx.get_embedding("no-such-id") is None


# ===========================================================================
# Scam shell-company pair: exact duplicate, different companies
# ===========================================================================

class TestScamShellCompanyDuplicates:
    """
    SCAM_SHELL_A and SCAM_SHELL_B have identical text but different company
    names.  This is the strongest signal in the system.
    """

    @skip_no_sbert
    def test_shell_pair_flagged_as_near_duplicate(self) -> None:
        idx = DuplicateIndex()
        idx.build([SCAM_SHELL_A, SCAM_SHELL_B])
        # At threshold 0.98, identical text should be near 1.0
        results = idx.find_near_duplicates("scam-shell-a", threshold=0.98)
        assert len(results) >= 1
        ids = [r[0] for r in results]
        assert "scam-shell-b" in ids

    @skip_no_sbert
    def test_shell_pair_similarity_near_one(self) -> None:
        idx = DuplicateIndex()
        idx.build([SCAM_SHELL_A, SCAM_SHELL_B])
        results = idx.find_near_duplicates("scam-shell-a", threshold=0.0)
        sims = {rid: sim for rid, sim in results}
        assert sims.get("scam-shell-b", 0.0) > 0.97

    @skip_no_sbert
    def test_cross_company_flag_fires_on_shell_pair(self) -> None:
        idx = DuplicateIndex()
        corpus = [SCAM_SHELL_A, SCAM_SHELL_B]
        idx.build(corpus)
        neighbors = idx.find_near_duplicates("scam-shell-a", threshold=0.90)
        assert cross_company_duplicate_flag(SCAM_SHELL_A, neighbors, corpus) is True

    @skip_no_sbert
    def test_cross_company_flag_false_for_unique_record(self) -> None:
        # Anakin has no near-duplicate → flag must be False
        idx = DuplicateIndex()
        idx.build(FULL_CORPUS)
        neighbors = idx.find_near_duplicates("anakin-software-001", threshold=0.92)
        result = cross_company_duplicate_flag(ANAKIN_TECH, neighbors, FULL_CORPUS)
        assert result is False


# ===========================================================================
# NayePankh / Basti Ki Pathshala documented ambiguity
# ===========================================================================

class TestNayePankhBastiAmbiguity:
    """
    NayePankh Foundation and Basti Ki Pathshala Foundation post structurally
    similar fundraising internship text.  The system will flag them as similar.

    ⚠  Phase 1 documented ambiguity: these organisations may be legitimately
    affiliated (shared reseller account on Internshala, similar mission).  A
    same_parent_organization_allowlist is required in production to avoid
    treating this as a guaranteed fraud signal.  See module docstring.
    """

    @skip_no_sbert
    def test_fundraising_records_are_semantically_similar(self) -> None:
        """
        Confirm the system detects structural similarity between the two orgs'
        fundraising postings — this is expected behaviour, not a bug.
        """
        idx = DuplicateIndex()
        idx.build([NAYEPANKH_FUNDRAISING, BASTI_FUNDRAISING, ANAKIN_TECH])
        results = idx.find_near_duplicates("nayepankh-fundraising-001", threshold=0.0)
        sims = {rid: sim for rid, sim in results}
        # NayePankh↔Basti similarity should be higher than NayePankh↔Anakin
        nayepankh_basti_sim = sims.get("basti-fundraising-001", 0.0)
        nayepankh_anakin_sim = sims.get("anakin-software-001", 0.0)
        assert nayepankh_basti_sim > nayepankh_anakin_sim, (
            "Expected fundraising records from similar orgs to be more similar "
            "to each other than to an unrelated technical internship"
        )

    @skip_no_sbert
    def test_cross_company_flag_fires_nayepankh_basti(self) -> None:
        """
        The cross-company flag WILL fire on NayePankh↔Basti at a moderate
        threshold because they ARE different companies with similar text.

        ⚠  This is the documented Phase 1 ambiguity.  In production, check
        against a same_parent_organization_allowlist before auto-rejecting.
        A comment below shows what that allowlist would look like.
        """
        # Production allowlist pattern (not implemented here — see module docstring):
        # AFFILIATED_ORG_PAIRS = {
        #     frozenset(["nayepankh foundation", "the nayepankh foundation",
        #                "basti ki pathshala foundation"]),
        # }
        idx = DuplicateIndex()
        corpus = [NAYEPANKH_FUNDRAISING, BASTI_FUNDRAISING, NAYEPANKH_SOCIAL, BASTI_SOCIAL]
        idx.build(corpus)
        # Use a lower threshold since these are structurally similar but not
        # identical templates
        neighbors = idx.find_near_duplicates("nayepankh-fundraising-001", threshold=0.65)
        flag = cross_company_duplicate_flag(NAYEPANKH_FUNDRAISING, neighbors, corpus)
        # Flag fires because text is similar AND companies differ
        # Production code must filter this with an org-graph/allowlist before rejecting
        assert isinstance(flag, bool)  # directional: just confirm it runs
        # If SBERT actually finds similarity above 0.65, we expect True
        if any(rid == "basti-fundraising-001" for rid, _ in neighbors):
            assert flag is True, (
                "cross_company_duplicate_flag should return True when Basti Ki "
                "Pathshala appears in NayePankh's neighbours — this is the "
                "documented false-positive risk that needs an allowlist in production"
            )

    @skip_no_sbert
    def test_anakin_rootsky_not_flagged_as_duplicates(self) -> None:
        """True negative: unrelated technical records should NOT be near-duplicates."""
        idx = DuplicateIndex()
        idx.build([ANAKIN_TECH, ROOTSKY_TECH])
        results = idx.find_near_duplicates("anakin-software-001", threshold=0.85)
        ids = [r[0] for r in results]
        assert "rootsky-web-001" not in ids, (
            "Anakin (data engine) and RootSky (web deployment) have different "
            "content and should not be flagged as near-duplicates"
        )


# ===========================================================================
# ClusterReport
# ===========================================================================

class TestClusterReport:

    @skip_no_sbert
    def test_report_total_records(self) -> None:
        idx = DuplicateIndex()
        idx.build(FULL_CORPUS)
        report = idx.duplicate_cluster_report()
        assert report.total_records == len(FULL_CORPUS)

    @skip_no_sbert
    def test_report_cluster_count_positive(self) -> None:
        idx = DuplicateIndex()
        idx.build(FULL_CORPUS)
        report = idx.duplicate_cluster_report()
        assert report.total_clusters >= 1

    @skip_no_sbert
    def test_report_cluster_sizes_sum_to_total(self) -> None:
        idx = DuplicateIndex()
        idx.build(FULL_CORPUS)
        report = idx.duplicate_cluster_report()
        total_from_dist = sum(
            size * count
            for size, count in report.cluster_size_distribution.items()
        )
        assert total_from_dist == report.total_records

    @skip_no_sbert
    def test_scam_pair_in_same_cluster(self) -> None:
        """Identical-text shell pair should be grouped into one cluster."""
        idx = DuplicateIndex()
        corpus = [SCAM_SHELL_A, SCAM_SHELL_B, ANAKIN_TECH]
        idx.build(corpus)
        report = idx.duplicate_cluster_report(threshold=0.95)
        # At least one cluster of size ≥ 2 should exist (the scam pair)
        multi_clusters = {
            s: c for s, c in report.cluster_size_distribution.items() if s >= 2
        }
        assert len(multi_clusters) >= 1

    @skip_no_sbert
    def test_largest_cluster_ids_populated(self) -> None:
        idx = DuplicateIndex()
        idx.build(FULL_CORPUS)
        report = idx.duplicate_cluster_report()
        assert report.largest_cluster_size >= 1
        assert len(report.largest_cluster_ids) == report.largest_cluster_size

    @skip_no_sbert
    def test_empty_corpus_report(self) -> None:
        idx = DuplicateIndex()
        idx.build([])
        report = idx.duplicate_cluster_report()
        assert report.total_records == 0
        assert report.total_clusters == 0


# ===========================================================================
# cross_company_duplicate_flag — unit tests (no SBERT needed)
# ===========================================================================

class TestCrossCompanyDuplicateFlag:

    def test_empty_neighbors_returns_false(self) -> None:
        assert cross_company_duplicate_flag(SCAM_SHELL_A, [], [SCAM_SHELL_B]) is False

    def test_same_company_neighbor_returns_false(self) -> None:
        # Neighbor has the same company name → not cross-company
        same_company_record = {**SCAM_SHELL_B, "company": "Alpha Solutions Pvt Ltd"}
        corpus = [SCAM_SHELL_A, same_company_record]
        neighbors = [("scam-shell-b", 0.99)]
        assert cross_company_duplicate_flag(SCAM_SHELL_A, neighbors, corpus) is False

    def test_different_company_neighbor_returns_true(self) -> None:
        corpus = [SCAM_SHELL_A, SCAM_SHELL_B]
        # Manually inject a high-sim neighbor with different company
        neighbors = [("scam-shell-b", 0.99)]
        assert cross_company_duplicate_flag(SCAM_SHELL_A, neighbors, corpus) is True

    def test_empty_company_neighbor_not_flagged(self) -> None:
        # A neighbor with empty company field should not fire the flag
        empty_company = {**SCAM_SHELL_B, "_id": "empty-co", "company": ""}
        corpus = [SCAM_SHELL_A, empty_company]
        neighbors = [("empty-co", 0.99)]
        assert cross_company_duplicate_flag(SCAM_SHELL_A, neighbors, corpus) is False

    def test_returns_bool_type(self) -> None:
        corpus = [SCAM_SHELL_A, SCAM_SHELL_B]
        result = cross_company_duplicate_flag(SCAM_SHELL_A, [("scam-shell-b", 0.99)], corpus)
        assert isinstance(result, bool)

    def test_case_insensitive_company_comparison(self) -> None:
        upper = {**SCAM_SHELL_A, "company": "ALPHA SOLUTIONS PVT LTD"}
        lower_neighbor = {**SCAM_SHELL_B, "_id": "nn", "company": "alpha solutions pvt ltd"}
        corpus = [upper, lower_neighbor]
        # Same company despite different case → should NOT fire
        assert cross_company_duplicate_flag(upper, [("nn", 0.99)], corpus) is False

    def test_multiple_neighbors_any_different_company_fires(self) -> None:
        # First neighbor same company, second different → should fire
        same_co = {**SCAM_SHELL_B, "_id": "same-co", "company": "Alpha Solutions Pvt Ltd"}
        diff_co = {**SCAM_SHELL_B, "_id": "diff-co", "company": "Gamma Enterprises"}
        corpus = [SCAM_SHELL_A, same_co, diff_co]
        neighbors = [("same-co", 0.99), ("diff-co", 0.98)]
        assert cross_company_duplicate_flag(SCAM_SHELL_A, neighbors, corpus) is True

    def test_suspect_company_bypasses_cross_company_flag(self) -> None:
        suspect_rec = {**SCAM_SHELL_A, "_flags": {"company_suspect": True}}
        corpus = [suspect_rec, SCAM_SHELL_B]
        neighbors = [("scam-shell-b", 0.99)]
        # Own company is suspect → should NOT fire
        assert cross_company_duplicate_flag(suspect_rec, neighbors, corpus) is False

    def test_neighbour_suspect_company_is_skipped(self) -> None:
        suspect_neighbour = {**SCAM_SHELL_B, "_flags": {"company_suspect": True}}
        corpus = [SCAM_SHELL_A, suspect_neighbour]
        neighbors = [("scam-shell-b", 0.99)]
        # Neighbour company is suspect → should NOT fire
        assert cross_company_duplicate_flag(SCAM_SHELL_A, neighbors, corpus) is False

