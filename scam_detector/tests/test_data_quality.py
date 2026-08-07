"""
test_data_quality.py
--------------------
Tests for scam_detector.data_quality.remediate

Fixture data is either hardcoded to match the exact patterns observed in the
corpus (Internshala / Naukri scraper output) or constructed minimally to
exercise a single code path.

Real-data patterns confirmed in:
  internScraper/scrapers/intershala_scraper/internships.json  — company leaks
  internScraper/scrapers/letsintern_scraper/internships.json  — company leaks
  naukri_internships.json                                     — null deadline
  internScraper/scrapers/intershala_scraper/internships.json  — trailing "4."
"""

from __future__ import annotations

import pytest

from scam_detector.data_quality import (
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

# ===========================================================================
# Shared fixtures
# ===========================================================================

# Real pattern: Internshala scraper sets company = the job-category string.
# Observed for hundreds of records (Digital Marketing, MS-Office, etc.)
FIXTURE_COMPANY_CATEGORY_LEAK = {
    "name": "Data Science Internship",
    "company": "Digital Marketing",
    "applyLink": "https://letsintern.in/data-science/",
    "summary": "Work on real data science projects with live datasets.",
    "skills": ["Python", "Pandas", "Machine Learning"],
    "tags": ["internship", "remote"],
    "deadlineDate": None,
    "datePublished": "2026-05-01",
    "createdAt": "2026-05-10",
    "responsibilities": [],
    "degree": None,
}

# company matches a value in the record's own skills list
FIXTURE_COMPANY_IN_SKILLS = {
    "name": "Python Developer Intern",
    "company": "Python",
    "applyLink": "https://example.com/apply",
    "summary": "Work with our Python backend team.",
    "skills": ["Python", "Django", "REST"],
    "tags": [],
    "deadlineDate": "2026-08-01",
    "datePublished": "2026-05-01",
    "createdAt": "2026-05-10",
    "responsibilities": [],
    "degree": None,
}

# company matches a value in the record's own tags list
FIXTURE_COMPANY_IN_TAGS = {
    "name": "Marketing Intern",
    "company": "remote",
    "applyLink": "https://example.com/apply",
    "summary": "Remote marketing role.",
    "skills": [],
    "tags": ["remote", "internship"],
    "deadlineDate": "2026-08-01",
    "datePublished": "2026-05-01",
    "createdAt": "2026-05-10",
    "responsibilities": [],
    "degree": None,
}

# Real pattern: degree = ["M.Tech"] + non-engineering title (Fundraising)
# Observed: Internshala fundraising roles with B.Tech/M.Tech fallback degree.
FIXTURE_DEGREE_DEFAULT_NON_ENGINEERING = {
    "name": "Fundraising",
    "company": "NayePankh Foundation",
    "applyLink": "https://internshala.com/internship/detail/fundraising1777892293",
    "summary": (
        "Support planning and execution of fundraising initiatives. "
        "Build donor relations and outreach campaigns."
    ),
    "skills": [],
    "tags": ["internship", "remote"],
    "degree": ["M.Tech"],
    "deadlineDate": None,
    "datePublished": "2026-06-01",
    "createdAt": "2026-06-01",
    "responsibilities": [],
}

# degree = ["B.Tech", "M.Tech"] but role IS engineering — should NOT flag
FIXTURE_DEGREE_DEFAULT_ENGINEERING = {
    "name": "Software Engineer Intern",
    "company": "Anakin",
    "applyLink": "https://internshala.com/internship/detail/software-development1777439557",
    "summary": "Build backend services, APIs, and data pipelines. Work on software engineering challenges.",
    "skills": ["Java", "Python"],
    "tags": [],
    "degree": ["B.Tech", "M.Tech"],
    "deadlineDate": "2026-07-01",
    "datePublished": "2026-05-01",
    "createdAt": "2026-05-10",
    "responsibilities": [],
}

# degree is an unusual set — should NOT flag even if non-engineering
FIXTURE_DEGREE_UNUSUAL_SET = {
    "name": "Graphic Design Intern",
    "company": "Moledro",
    "applyLink": "https://internshala.com/internship/detail/branding1777870618",
    "summary": "Design social media creatives and brand assets.",
    "skills": ["Canva", "Photoshop"],
    "tags": [],
    "degree": ["BFA", "B.Des"],
    "deadlineDate": "2026-07-01",
    "datePublished": "2026-05-01",
    "createdAt": "2026-05-10",
    "responsibilities": [],
}

# Real pattern: Naukri scraper produces null deadlineDate for most records
FIXTURE_NULL_DEADLINE = {
    "name": "Intern",
    "company": "Naukri Employer",
    "applyLink": "https://www.naukri.com",
    "datePublished": {"$date": "2024-06-07T00:00:00Z"},
    "deadlineDate": None,
    "summary": "Great opportunity to start your career.",
    "skills": ["Communication", "Problem Solving"],
    "degree": ["B.E/B.Tech", "BCA", "Any Graduate"],
    "tags": ["internship", "fresher"],
    "createdAt": {"$date": "2024-06-07T00:00:00Z"},
    "responsibilities": [
        "Work on real-world projects and tasks",
        "Collaborate with team members to deliver results",
    ],
}

# Real pattern: Internshala responsibility strings end with stray "  4."
# e.g. "...identify new outreach opportunities   4."
FIXTURE_RESPONSIBILITY_TRAILING_ARTIFACT = {
    "name": "Business Consultant",
    "company": "NayePankh Foundation",
    "applyLink": "https://internshala.com/internship/detail/business-consultant1777892177",
    "summary": "Hands-on exposure to performance analysis and strategic planning.",
    "skills": [],
    "tags": [],
    "degree": None,
    "deadlineDate": "2026-06-30",
    "datePublished": "2026-05-01",
    "createdAt": "2026-05-10",
    "responsibilities": [
        # Clean items — should be untouched
        "Review and evaluate fundraising performance metrics and campaign outcomes",
        "Develop and recommend strategic improvements for fundraising teams and interns",
        # Items with trailing stray index — should be stripped
        "Identify operational gaps, process inefficiencies, and new outreach opportunities   4.",
        "Contribute to enhancing donor engagement and retention strategies   5.",
        # Two-digit index
        "Collaborate with core team members to support data-driven decision-making   12.",
    ],
}

# summary ends with "..."
FIXTURE_TRUNCATED_SUMMARY_ELLIPSIS = {
    "name": "Marketing Intern",
    "company": "SomeStartup",
    "applyLink": "https://example.com/apply",
    "summary": "We are looking for a dynamic marketing intern who will help us grow our brand...",
    "skills": [],
    "tags": [],
    "degree": None,
    "deadlineDate": "2026-07-01",
    "datePublished": "2026-05-01",
    "createdAt": "2026-05-10",
    "responsibilities": [],
}

# summary is exactly 300 chars (observed Internshala truncation)
FIXTURE_TRUNCATED_SUMMARY_300 = {
    "name": "Digital Marketing Intern",
    "company": "SomeCo",
    "applyLink": "https://example.com/apply",
    "summary": "A" * 300,
    "skills": [],
    "tags": [],
    "degree": None,
    "deadlineDate": "2026-07-01",
    "datePublished": "2026-05-01",
    "createdAt": "2026-05-10",
    "responsibilities": [],
}

# datePublished == createdAt (same calendar day) → date was likely inferred
FIXTURE_DATE_INFERRED = {
    "name": "Intern",
    "company": "SomeCo",
    "applyLink": "https://example.com/apply",
    "summary": "Short internship.",
    "skills": [],
    "tags": [],
    "degree": None,
    "deadlineDate": "2026-08-01",
    "datePublished": "2026-06-15T09:23:11",
    "createdAt": "2026-06-15T14:00:00",   # same calendar date, different time
    "responsibilities": [],
}

# datePublished != createdAt → not flagged
FIXTURE_DATE_NOT_INFERRED = {
    "name": "Intern",
    "company": "SomeCo",
    "applyLink": "https://example.com/apply",
    "summary": "Short internship.",
    "skills": [],
    "tags": [],
    "degree": None,
    "deadlineDate": "2026-08-01",
    "datePublished": "2026-06-10",
    "createdAt": "2026-06-15",
    "responsibilities": [],
}

# MongoDB Extended JSON date format
FIXTURE_DATE_INFERRED_MONGO = {
    "name": "Intern",
    "company": "Naukri Employer",
    "applyLink": "https://www.naukri.com",
    "summary": "Great opportunity.",
    "skills": [],
    "tags": [],
    "degree": None,
    "deadlineDate": None,
    "datePublished": {"$date": "2024-06-07T00:00:00Z"},
    "createdAt": {"$date": "2024-06-07T00:00:00Z"},
    "responsibilities": [],
}


# ===========================================================================
# Fix 1 — flag_mislabeled_company
# ===========================================================================

class TestFlagMislabeledCompany:

    def test_hardcoded_term_detected(self) -> None:
        _, flags = flag_mislabeled_company(FIXTURE_COMPANY_CATEGORY_LEAK)
        assert flags.get("company_suspect") is True
        assert flags.get("company_source") == "category_leak"
        assert flags.get("company_match_via") == "hardcoded_list"

    def test_company_in_skills_detected(self) -> None:
        _, flags = flag_mislabeled_company(FIXTURE_COMPANY_IN_SKILLS)
        assert flags.get("company_suspect") is True
        assert flags.get("company_match_via") == "skills_field"

    def test_company_in_tags_detected(self) -> None:
        _, flags = flag_mislabeled_company(FIXTURE_COMPANY_IN_TAGS)
        assert flags.get("company_suspect") is True
        assert flags.get("company_match_via") == "tags_field"

    def test_real_company_not_flagged(self) -> None:
        _, flags = flag_mislabeled_company(FIXTURE_DEGREE_DEFAULT_NON_ENGINEERING)
        assert "company_suspect" not in flags

    def test_case_insensitive(self) -> None:
        record = {**FIXTURE_COMPANY_CATEGORY_LEAK, "company": "DIGITAL MARKETING"}
        _, flags = flag_mislabeled_company(record)
        assert flags.get("company_suspect") is True

    def test_company_value_not_mutated(self) -> None:
        original_company = FIXTURE_COMPANY_CATEGORY_LEAK["company"]
        returned_record, _ = flag_mislabeled_company(FIXTURE_COMPANY_CATEGORY_LEAK)
        assert returned_record["company"] == original_company

    def test_ms_office_detected(self) -> None:
        record = {**FIXTURE_NULL_DEADLINE, "company": "MS-Office"}
        _, flags = flag_mislabeled_company(record)
        assert flags.get("company_suspect") is True

    def test_whatsapp_detected(self) -> None:
        record = {**FIXTURE_NULL_DEADLINE, "company": "WhatsApp"}
        _, flags = flag_mislabeled_company(record)
        assert flags.get("company_suspect") is True

    def test_empty_company_no_flag(self) -> None:
        record = {**FIXTURE_NULL_DEADLINE, "company": ""}
        _, flags = flag_mislabeled_company(record)
        assert "company_suspect" not in flags

    def test_social_media_marketing_detected(self) -> None:
        record = {**FIXTURE_NULL_DEADLINE, "company": "Social Media Marketing"}
        _, flags = flag_mislabeled_company(record)
        assert flags.get("company_suspect") is True


# ===========================================================================
# Fix 2 — flag_degree_default
# ===========================================================================

class TestFlagDegreeDefault:

    def test_mtech_only_non_engineering_flagged(self) -> None:
        _, flags = flag_degree_default(FIXTURE_DEGREE_DEFAULT_NON_ENGINEERING)
        assert flags.get("degree_suspect_default") is True

    def test_btech_mtech_non_engineering_flagged(self) -> None:
        record = {**FIXTURE_DEGREE_DEFAULT_NON_ENGINEERING, "degree": ["B.Tech", "M.Tech"]}
        _, flags = flag_degree_default(record)
        assert flags.get("degree_suspect_default") is True

    def test_engineering_role_not_flagged(self) -> None:
        _, flags = flag_degree_default(FIXTURE_DEGREE_DEFAULT_ENGINEERING)
        assert "degree_suspect_default" not in flags

    def test_unusual_degree_set_not_flagged(self) -> None:
        _, flags = flag_degree_default(FIXTURE_DEGREE_UNUSUAL_SET)
        assert "degree_suspect_default" not in flags

    def test_no_degree_field_not_flagged(self) -> None:
        record = {**FIXTURE_DEGREE_DEFAULT_NON_ENGINEERING, "degree": None}
        _, flags = flag_degree_default(record)
        assert "degree_suspect_default" not in flags

    def test_empty_degree_list_not_flagged(self) -> None:
        record = {**FIXTURE_DEGREE_DEFAULT_NON_ENGINEERING, "degree": []}
        _, flags = flag_degree_default(record)
        assert "degree_suspect_default" not in flags

    def test_engineering_keyword_in_summary_saves_record(self) -> None:
        # M.Tech + summary mentions "software" → should NOT flag
        record = {
            **FIXTURE_DEGREE_DEFAULT_NON_ENGINEERING,
            "summary": "Work on software systems and developer tooling.",
        }
        _, flags = flag_degree_default(record)
        assert "degree_suspect_default" not in flags

    def test_engineering_keyword_in_title_saves_record(self) -> None:
        record = {
            **FIXTURE_DEGREE_DEFAULT_NON_ENGINEERING,
            "name": "Technical Support Intern",
        }
        _, flags = flag_degree_default(record)
        assert "degree_suspect_default" not in flags

    def test_order_independence(self) -> None:
        # ["M.Tech", "B.Tech"] and ["B.Tech", "M.Tech"] should behave identically
        record_a = {**FIXTURE_DEGREE_DEFAULT_NON_ENGINEERING, "degree": ["M.Tech", "B.Tech"]}
        record_b = {**FIXTURE_DEGREE_DEFAULT_NON_ENGINEERING, "degree": ["B.Tech", "M.Tech"]}
        _, flags_a = flag_degree_default(record_a)
        _, flags_b = flag_degree_default(record_b)
        assert flags_a == flags_b


# ===========================================================================
# Fix 3 — flag_missing_deadline
# ===========================================================================

class TestFlagMissingDeadline:

    def test_null_deadline_flagged(self) -> None:
        _, flags = flag_missing_deadline(FIXTURE_NULL_DEADLINE)
        assert flags.get("deadline_missing") is True

    def test_missing_key_flagged(self) -> None:
        record = {k: v for k, v in FIXTURE_NULL_DEADLINE.items() if k != "deadlineDate"}
        _, flags = flag_missing_deadline(record)
        assert flags.get("deadline_missing") is True

    def test_empty_string_deadline_flagged(self) -> None:
        record = {**FIXTURE_NULL_DEADLINE, "deadlineDate": ""}
        _, flags = flag_missing_deadline(record)
        assert flags.get("deadline_missing") is True

    def test_real_deadline_not_flagged(self) -> None:
        _, flags = flag_missing_deadline(FIXTURE_DEGREE_DEFAULT_ENGINEERING)
        assert "deadline_missing" not in flags

    def test_record_not_mutated(self) -> None:
        original = dict(FIXTURE_NULL_DEADLINE)
        returned_record, _ = flag_missing_deadline(FIXTURE_NULL_DEADLINE)
        assert returned_record["deadlineDate"] == original["deadlineDate"]


# ===========================================================================
# Fix 4 — clean_responsibilities
# ===========================================================================

class TestCleanResponsibilities:

    def test_trailing_single_digit_stripped(self) -> None:
        record, flags = clean_responsibilities(FIXTURE_RESPONSIBILITY_TRAILING_ARTIFACT)
        assert flags.get("responsibilities_cleaned") is True
        for item in record["responsibilities"]:
            assert not item.endswith("4.")
            assert not item.endswith("5.")
            assert not item.endswith("12.")

    def test_two_digit_trailing_index_stripped(self) -> None:
        record = {
            **FIXTURE_NULL_DEADLINE,
            "responsibilities": ["Do something important   12."],
        }
        cleaned, flags = clean_responsibilities(record)
        assert flags.get("responsibilities_cleaned") is True
        assert cleaned["responsibilities"][0] == "Do something important"

    def test_clean_items_untouched(self) -> None:
        record, _ = clean_responsibilities(FIXTURE_RESPONSIBILITY_TRAILING_ARTIFACT)
        # First two items were already clean
        assert record["responsibilities"][0] == (
            "Review and evaluate fundraising performance metrics and campaign outcomes"
        )
        assert record["responsibilities"][1] == (
            "Develop and recommend strategic improvements for fundraising teams and interns"
        )

    def test_no_artifact_no_flag(self) -> None:
        record = {
            **FIXTURE_NULL_DEADLINE,
            "responsibilities": [
                "Work on real-world projects and tasks",
                "Collaborate with team members to deliver results.",
            ],
        }
        _, flags = clean_responsibilities(record)
        assert "responsibilities_cleaned" not in flags

    def test_empty_responsibilities_no_flag(self) -> None:
        record = {**FIXTURE_NULL_DEADLINE, "responsibilities": []}
        _, flags = clean_responsibilities(record)
        assert "responsibilities_cleaned" not in flags

    def test_none_responsibilities_no_flag(self) -> None:
        record = {**FIXTURE_NULL_DEADLINE, "responsibilities": None}
        _, flags = clean_responsibilities(record)
        assert "responsibilities_cleaned" not in flags

    def test_legitimate_sentence_ending_period_preserved(self) -> None:
        # "...make a real impact." — ends in a word + period, no leading digits
        record = {
            **FIXTURE_NULL_DEADLINE,
            "responsibilities": ["Join us in making a difference today!"],
        }
        cleaned, flags = clean_responsibilities(record)
        assert "responsibilities_cleaned" not in flags
        assert cleaned["responsibilities"][0] == "Join us in making a difference today!"

    def test_original_record_not_mutated(self) -> None:
        original_resp = list(FIXTURE_RESPONSIBILITY_TRAILING_ARTIFACT["responsibilities"])
        clean_responsibilities(FIXTURE_RESPONSIBILITY_TRAILING_ARTIFACT)
        assert FIXTURE_RESPONSIBILITY_TRAILING_ARTIFACT["responsibilities"] == original_resp

    def test_whitespace_stripped_around_artifact(self) -> None:
        record = {
            **FIXTURE_NULL_DEADLINE,
            "responsibilities": ["Do the thing   6.  "],
        }
        # The regex matches \s+\d{1,2}\.\s*$ — trailing spaces after "6." also gone
        cleaned, _ = clean_responsibilities(record)
        assert cleaned["responsibilities"][0].endswith("Do the thing")


# ===========================================================================
# Fix 5 — flag_truncated_summary
# ===========================================================================

class TestFlagTruncatedSummary:

    def test_ellipsis_suffix_flagged(self) -> None:
        _, flags = flag_truncated_summary(FIXTURE_TRUNCATED_SUMMARY_ELLIPSIS)
        assert flags.get("summary_truncated") is True

    def test_exactly_300_chars_flagged(self) -> None:
        _, flags = flag_truncated_summary(FIXTURE_TRUNCATED_SUMMARY_300)
        assert flags.get("summary_truncated") is True

    def test_unicode_ellipsis_flagged(self) -> None:
        record = {**FIXTURE_TRUNCATED_SUMMARY_ELLIPSIS, "summary": "Some description\u2026"}
        _, flags = flag_truncated_summary(record)
        assert flags.get("summary_truncated") is True

    def test_normal_summary_not_flagged(self) -> None:
        _, flags = flag_truncated_summary(FIXTURE_NULL_DEADLINE)
        assert "summary_truncated" not in flags

    def test_299_chars_not_flagged(self) -> None:
        record = {**FIXTURE_TRUNCATED_SUMMARY_300, "summary": "A" * 299}
        _, flags = flag_truncated_summary(record)
        assert "summary_truncated" not in flags

    def test_301_chars_not_flagged(self) -> None:
        record = {**FIXTURE_TRUNCATED_SUMMARY_300, "summary": "A" * 301}
        _, flags = flag_truncated_summary(record)
        assert "summary_truncated" not in flags

    def test_empty_summary_no_flag(self) -> None:
        record = {**FIXTURE_TRUNCATED_SUMMARY_300, "summary": ""}
        _, flags = flag_truncated_summary(record)
        assert "summary_truncated" not in flags


# ===========================================================================
# Fix 6 — flag_inferred_date
# ===========================================================================

class TestFlagInferredDate:

    def test_same_day_iso_string_flagged(self) -> None:
        _, flags = flag_inferred_date(FIXTURE_DATE_INFERRED)
        assert flags.get("date_possibly_inferred") is True

    def test_different_days_not_flagged(self) -> None:
        _, flags = flag_inferred_date(FIXTURE_DATE_NOT_INFERRED)
        assert "date_possibly_inferred" not in flags

    def test_mongo_extended_json_same_day_flagged(self) -> None:
        _, flags = flag_inferred_date(FIXTURE_DATE_INFERRED_MONGO)
        assert flags.get("date_possibly_inferred") is True

    def test_missing_created_at_not_flagged(self) -> None:
        record = {k: v for k, v in FIXTURE_DATE_INFERRED.items() if k != "createdAt"}
        _, flags = flag_inferred_date(record)
        assert "date_possibly_inferred" not in flags

    def test_missing_date_published_not_flagged(self) -> None:
        record = {k: v for k, v in FIXTURE_DATE_INFERRED.items() if k != "datePublished"}
        _, flags = flag_inferred_date(record)
        assert "date_possibly_inferred" not in flags

    def test_same_date_different_times_still_flagged(self) -> None:
        # Both on 2026-06-15 — different wall-clock times but same calendar date
        record = {
            **FIXTURE_DATE_INFERRED,
            "datePublished": "2026-06-15T00:00:00",
            "createdAt": "2026-06-15T23:59:59",
        }
        _, flags = flag_inferred_date(record)
        assert flags.get("date_possibly_inferred") is True


# ===========================================================================
# remediate_record — integration
# ===========================================================================

class TestRemediateRecord:

    def test_returns_remediated_record_type(self) -> None:
        result = remediate_record(FIXTURE_COMPANY_CATEGORY_LEAK)
        assert isinstance(result, RemediatedRecord)

    def test_flags_are_dict(self) -> None:
        result = remediate_record(FIXTURE_COMPANY_CATEGORY_LEAK)
        assert isinstance(result.flags, dict)

    def test_all_six_passes_run(self) -> None:
        # Construct a record that triggers five of the six flags simultaneously
        record = {
            "name": "Fundraising",
            "company": "Digital Marketing",          # → company_suspect
            "applyLink": "https://example.com/apply",
            "summary": "Short description...",        # → summary_truncated
            "skills": [],
            "tags": [],
            "degree": ["M.Tech"],                    # → degree_suspect_default
            "deadlineDate": None,                    # → deadline_missing
            "datePublished": "2026-06-01",           # same day → date_possibly_inferred
            "createdAt": "2026-06-01",
            "responsibilities": [
                "Do important work   3.",            # → responsibilities_cleaned
            ],
        }
        result = remediate_record(record)
        assert result.flags.get("company_suspect") is True
        assert result.flags.get("degree_suspect_default") is True
        assert result.flags.get("deadline_missing") is True
        assert result.flags.get("responsibilities_cleaned") is True
        assert result.flags.get("summary_truncated") is True
        assert result.flags.get("date_possibly_inferred") is True

    def test_original_dict_not_mutated(self) -> None:
        original = dict(FIXTURE_RESPONSIBILITY_TRAILING_ARTIFACT)
        original_resp = list(original["responsibilities"])
        remediate_record(FIXTURE_RESPONSIBILITY_TRAILING_ARTIFACT)
        assert FIXTURE_RESPONSIBILITY_TRAILING_ARTIFACT["responsibilities"] == original_resp

    def test_clean_record_produces_no_flags(self) -> None:
        clean = {
            "name": "Software Engineer Intern",
            "company": "Razorpay",
            "applyLink": "https://razorpay.com/jobs/intern",
            "summary": "Work with Razorpay's frontend team to build payment dashboard components used by millions of merchants across India.",
            "skills": ["React", "TypeScript"],
            "tags": ["frontend", "fintech"],
            "degree": ["B.Tech", "B.E.", "BCA"],
            "deadlineDate": "2026-07-05",
            "datePublished": "2026-06-05",
            "createdAt": "2026-06-06",
            "responsibilities": [
                "Develop responsive UI components using React and TypeScript",
                "Integrate REST APIs for live payment data",
            ],
        }
        result = remediate_record(clean)
        assert result.flags == {}

    def test_empty_dict_does_not_raise(self) -> None:
        result = remediate_record({})
        assert isinstance(result, RemediatedRecord)

    def test_record_field_preserved_on_result(self) -> None:
        result = remediate_record(FIXTURE_NULL_DEADLINE)
        assert result.record["company"] == FIXTURE_NULL_DEADLINE["company"]


# ===========================================================================
# remediate_batch
# ===========================================================================

class TestRemediateBatch:

    def test_returns_list_of_remediated_records(self) -> None:
        batch = [FIXTURE_COMPANY_CATEGORY_LEAK, FIXTURE_NULL_DEADLINE, FIXTURE_DATE_INFERRED]
        results = remediate_batch(batch)
        assert len(results) == 3
        assert all(isinstance(r, RemediatedRecord) for r in results)

    def test_empty_batch(self) -> None:
        assert remediate_batch([]) == []

    def test_each_record_independently_remediated(self) -> None:
        results = remediate_batch([FIXTURE_COMPANY_CATEGORY_LEAK, FIXTURE_NULL_DEADLINE])
        assert results[0].flags.get("company_suspect") is True
        assert results[1].flags.get("deadline_missing") is True

    def test_error_in_one_record_raises_with_index(self) -> None:
        # Force a TypeError by passing a non-dict
        with pytest.raises(RuntimeError, match="index 1"):
            remediate_batch([FIXTURE_NULL_DEADLINE, "not a dict"])  # type: ignore[list-item]
