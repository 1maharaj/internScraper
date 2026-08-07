"""
test_stipend_temporal_structural.py
------------------------------------
Tests for:
  scam_detector.features.stipend_features
  scam_detector.features.temporal_features
  scam_detector.features.structural_features

Real corpus fixtures from:
  internScraper/checkpoint_internships.json:
    Anakin          — paid ₹20,000/month, 6 months, Software Engineering
    Moledro         — paid ₹10,001/month, 2 months, Branding
    Meritshot       — unpaid, 6 months, Software Development
    RootSky         — paid ₹7,000/month,  3 months, Software Engineering
    Edith Defence   — paid ₹12,000/month, 6 months, Computer Vision
"""
from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from scam_detector.features.stipend_features import (
    normalize_stipend_to_hourly_inr,
    stipend_zscore,
    stipend_perk_consistency_check,
    StipendFeatures,
)
from scam_detector.features.temporal_features import (
    posting_burst_score,
    deadline_urgency_score,
    TemporalFeatures,
    _as_date,
)
from scam_detector.features.structural_features import (
    openings_zscore,
    field_completeness_score,
    StructuralFeatures,
)

# ---------------------------------------------------------------------------
# Shared real-data fixtures
# ---------------------------------------------------------------------------

STIPEND_PAID_20K = {"type": "paid", "amount": 20000, "currency": "INR", "period": "monthly"}
STIPEND_PAID_10K = {"type": "paid", "amount": 10001, "currency": "INR", "period": "monthly"}
STIPEND_PAID_7K  = {"type": "paid", "amount": 7000,  "currency": "INR", "period": "monthly"}
STIPEND_PAID_12K = {"type": "paid", "amount": 12000, "currency": "INR", "period": "monthly"}
STIPEND_UNPAID   = {"type": "unpaid", "amount": None, "currency": "INR", "period": None}
STIPEND_PERF     = {"type": "performance-based", "amount": None, "currency": "INR", "period": None}
STIPEND_LUMPSUM  = {"type": "paid", "amount": 50000, "currency": "INR", "period": "lump-sum"}
STIPEND_USD      = {"type": "paid", "amount": 1000,  "currency": "USD", "period": "monthly"}

DUR_6M = {"value": 6, "unit": "months"}
DUR_3M = {"value": 3, "unit": "months"}
DUR_2M = {"value": 2, "unit": "months"}
DUR_0  = {"value": 0, "unit": "months"}   # edge case: zero duration

ANAKIN_RECORD = {
    "name": "Software Development",
    "company": "Anakin",
    "applyLink": "https://internshala.com/internship/detail/software-development1777439557",
    "stipend": STIPEND_PAID_20K,
    "duration": DUR_6M,
    "openings": 4,
    "field": ["Software Engineering", "Backend"],
    "tags": ["internship", "backend"],
    "skills": ["Java", "Python", "Algorithms"],
    "responsibilities": ["Build APIs", "Debug issues"],
    "perks": ["Certificate", "Letter of recommendation"],
    "degree": ["B.Tech"],
    "city": "Bangalore",
    "summary": "Large-scale data engine engineering internship.",
    "isRemote": False,
    "datePublished": "2026-05-01",
    "deadlineDate": "2026-05-29",
}

MOLEDRO_RECORD = {
    "name": "Branding/Social Media",
    "company": "Moledro",
    "applyLink": "https://internshala.com/internship/detail/branding1777870618",
    "stipend": STIPEND_PAID_10K,
    "duration": DUR_2M,
    "openings": 1,
    "field": ["Marketing", "Design"],
    "tags": ["internship", "branding"],
    "skills": ["Canva", "Adobe Creative Suite"],
    "responsibilities": ["Brainstorm content", "Develop copy"],
    "perks": ["Certificate"],
    "degree": None,
    "city": "Delhi",
    "summary": "Clothing brand seeking branding intern.",
    "isRemote": False,
    "datePublished": "2026-05-10",
    "deadlineDate": "2026-06-03",
}

MERITSHOT_RECORD = {
    "name": "Software Development Intern",
    "company": "Meritshot",
    "applyLink": "https://internshala.com/internship/detail/software-development-intern1777970950",
    "stipend": STIPEND_UNPAID,
    "duration": DUR_6M,
    "openings": 1,
    "field": ["Software Engineering"],
    "tags": ["internship"],
    "skills": [],
    "responsibilities": ["Design, code, test software"],
    "perks": [],
    "degree": ["B.Tech"],
    "city": "Noida",
    "summary": "Career transformation platform.",
    "isRemote": False,
    "datePublished": "2026-05-15",
    "deadlineDate": None,
}

ROOTSKY_RECORD = {
    "name": "Software Development Engineering (Web)",
    "company": "RootSky System (OPC) Private Limited",
    "applyLink": "https://internshala.com/internship/detail/software-development-engineering-web1777983872",
    "stipend": STIPEND_PAID_7K,
    "duration": DUR_3M,
    "openings": 2,
    "field": ["Software Engineering", "Web Development"],
    "tags": ["internship"],
    "skills": ["JavaScript", "PHP", "Linux"],
    "responsibilities": ["Environment setup", "Deployment"],
    "perks": ["Certificate", "Flexible work hours"],
    "degree": None,
    "city": "Noida",
    "summary": "Technology partner specializing in software development.",
    "isRemote": False,
    "datePublished": "2026-05-12",
    "deadlineDate": "2026-06-04",
}

EDITH_RECORD = {
    "name": "Computer Vision And Image Processing",
    "company": "Edith Defence Systems Private Limited",
    "applyLink": "https://internshala.com/internship/detail/computer-vision1777395711",
    "stipend": STIPEND_PAID_12K,
    "duration": DUR_6M,
    "openings": 2,
    "field": ["Computer Vision", "Machine Learning"],
    "tags": ["internship", "defence"],
    "skills": ["OpenCV", "Python", "C++", "Deep Learning"],
    "responsibilities": ["Design image processing techniques", "Develop CV modules"],
    "perks": ["Certificate", "Letter of recommendation", "Job offer"],
    "degree": ["B.Tech", "M.Tech"],
    "city": "Navi Mumbai",
    "summary": "AI/CV internship at defence company.",
    "isRemote": False,
    "datePublished": "2026-04-20",
    "deadlineDate": "2026-06-01",
}

# Peer group for stipend z-score tests (Software Engineering, on-site)
SWE_PEER_GROUP = [ANAKIN_RECORD, ROOTSKY_RECORD, EDITH_RECORD, MERITSHOT_RECORD]


# ===========================================================================
# normalize_stipend_to_hourly_inr
# ===========================================================================

class TestNormalizeStipendToHourlyInr:

    def test_monthly_inr_correct(self) -> None:
        # ₹20,000/month ÷ 160 hrs = ₹125/hr
        result = normalize_stipend_to_hourly_inr(STIPEND_PAID_20K, DUR_6M)
        assert result == pytest.approx(125.0, rel=1e-3)

    def test_weekly_inr_correct(self) -> None:
        stipend = {"type": "paid", "amount": 4000, "currency": "INR", "period": "weekly"}
        result = normalize_stipend_to_hourly_inr(stipend, DUR_3M)
        # ₹4,000/week ÷ 40 hrs = ₹100/hr
        assert result == pytest.approx(100.0, rel=1e-3)

    def test_lump_sum_amortised_over_duration(self) -> None:
        # ₹50,000 lump-sum over 6 months = ₹50,000 / (6×160) = ₹52.08/hr
        result = normalize_stipend_to_hourly_inr(STIPEND_LUMPSUM, DUR_6M)
        assert result == pytest.approx(50000 / (6 * 160), rel=1e-3)

    def test_unpaid_returns_zero(self) -> None:
        result = normalize_stipend_to_hourly_inr(STIPEND_UNPAID, DUR_6M)
        assert result == 0.0

    def test_performance_based_returns_none(self) -> None:
        result = normalize_stipend_to_hourly_inr(STIPEND_PERF, DUR_6M)
        assert result is None

    def test_usd_converted_to_inr(self) -> None:
        # $1,000/month × 83.5 FX ÷ 160 hrs ≈ ₹521.875/hr
        result = normalize_stipend_to_hourly_inr(STIPEND_USD, DUR_3M)
        assert result is not None
        assert result == pytest.approx(1000 * 83.5 / 160, rel=1e-2)

    def test_zero_duration_does_not_divide_by_zero(self) -> None:
        # lump-sum with zero duration → defaults to 1 month
        result = normalize_stipend_to_hourly_inr(STIPEND_LUMPSUM, DUR_0)
        assert result is not None
        assert math.isfinite(result)
        assert result == pytest.approx(50000 / 160, rel=1e-3)

    def test_missing_amount_returns_none(self) -> None:
        stipend = {"type": "paid", "amount": None, "currency": "INR", "period": "monthly"}
        assert normalize_stipend_to_hourly_inr(stipend, DUR_3M) is None

    def test_empty_stipend_returns_none(self) -> None:
        assert normalize_stipend_to_hourly_inr({}, DUR_3M) is None

    def test_none_stipend_returns_none(self) -> None:
        assert normalize_stipend_to_hourly_inr(None, DUR_3M) is None  # type: ignore[arg-type]

    def test_unknown_type_returns_none(self) -> None:
        stipend = {"type": "barter", "amount": 5000, "currency": "INR", "period": "monthly"}
        assert normalize_stipend_to_hourly_inr(stipend, DUR_3M) is None

    def test_unknown_currency_treated_as_inr(self) -> None:
        stipend = {"type": "paid", "amount": 10000, "currency": "XYZ", "period": "monthly"}
        result = normalize_stipend_to_hourly_inr(stipend, DUR_3M)
        # XYZ unknown → rate 1.0 → same as INR
        assert result == pytest.approx(10000 / 160, rel=1e-3)

    def test_weeks_duration_lump_sum(self) -> None:
        stipend = {"type": "paid", "amount": 20000, "currency": "INR", "period": "lump-sum"}
        duration = {"value": 8, "unit": "weeks"}
        result = normalize_stipend_to_hourly_inr(stipend, duration)
        assert result == pytest.approx(20000 / (8 * 40), rel=1e-3)


# ===========================================================================
# stipend_zscore
# ===========================================================================

class TestStipendZscore:

    def test_average_record_scores_near_zero(self) -> None:
        # EDITH: ₹12,000/month — middling in the SWE peer group
        z = stipend_zscore(EDITH_RECORD, SWE_PEER_GROUP)
        assert z is not None
        assert abs(z) < 2.0

    def test_highest_stipend_positive_z(self) -> None:
        # ANAKIN: ₹20,000 — highest in group → positive z
        z = stipend_zscore(ANAKIN_RECORD, SWE_PEER_GROUP)
        assert z is not None
        assert z > 0

    def test_lowest_stipend_negative_z(self) -> None:
        # ROOTSKY: ₹7,000 — lowest paid in group → negative z
        z = stipend_zscore(ROOTSKY_RECORD, SWE_PEER_GROUP)
        assert z is not None
        assert z < 0

    def test_performance_based_returns_none(self) -> None:
        record = {**ANAKIN_RECORD, "stipend": STIPEND_PERF}
        assert stipend_zscore(record, SWE_PEER_GROUP) is None

    def test_too_few_peers_returns_none(self) -> None:
        z = stipend_zscore(ANAKIN_RECORD, [ROOTSKY_RECORD])
        assert z is None

    def test_empty_peer_group_returns_none(self) -> None:
        assert stipend_zscore(ANAKIN_RECORD, []) is None

    def test_all_peers_identical_stipend_zero(self) -> None:
        same = {"type": "paid", "amount": 10000, "currency": "INR", "period": "monthly"}
        peer1 = {**ANAKIN_RECORD, "stipend": same}
        peer2 = {**ROOTSKY_RECORD, "stipend": same}
        record = {**EDITH_RECORD, "stipend": same}
        z = stipend_zscore(record, [peer1, peer2])
        assert z == 0.0

    def test_unpaid_included_in_peer_group(self) -> None:
        # Meritshot is unpaid (0 INR/hr), which counts as a valid peer value.
        # When added to a group of paid peers it changes both the mean and σ.
        # We just verify: (1) the function returns a float for Anakin in both
        # cases, and (2) the unpaid record's own z-score is negative (it is
        # the cheapest option in the peer group).
        z_anakin = stipend_zscore(ANAKIN_RECORD, SWE_PEER_GROUP)
        z_meritshot = stipend_zscore(MERITSHOT_RECORD, SWE_PEER_GROUP)
        assert z_anakin is not None and isinstance(z_anakin, float)
        # Unpaid (0) should be below the group mean → negative z
        assert z_meritshot is not None and z_meritshot < 0

    def test_return_is_float_when_computed(self) -> None:
        z = stipend_zscore(ANAKIN_RECORD, SWE_PEER_GROUP)
        assert isinstance(z, float)


# ===========================================================================
# stipend_perk_consistency_check
# ===========================================================================

class TestStipendPerkConsistencyCheck:

    def test_unpaid_with_stipend_perk_is_inconsistent(self) -> None:
        record = {**MERITSHOT_RECORD, "perks": ["Certificate", "Stipend"]}
        assert stipend_perk_consistency_check(record) is True

    def test_unpaid_with_monthly_stipend_perk(self) -> None:
        record = {**MERITSHOT_RECORD, "perks": ["Monthly Stipend", "Certificate"]}
        assert stipend_perk_consistency_check(record) is True

    def test_unpaid_no_stipend_perk_is_ok(self) -> None:
        assert stipend_perk_consistency_check(MERITSHOT_RECORD) is False

    def test_paid_with_stipend_perk_is_ok(self) -> None:
        record = {**ANAKIN_RECORD, "perks": ["Stipend", "Certificate"]}
        assert stipend_perk_consistency_check(record) is False

    def test_empty_perks_is_ok(self) -> None:
        record = {**MERITSHOT_RECORD, "perks": []}
        assert stipend_perk_consistency_check(record) is False

    def test_missing_stipend_is_ok(self) -> None:
        record = {"name": "Intern", "perks": ["Stipend"]}
        assert stipend_perk_consistency_check(record) is False

    def test_return_type_is_bool(self) -> None:
        assert isinstance(stipend_perk_consistency_check(MERITSHOT_RECORD), bool)

    def test_case_insensitive_perk_matching(self) -> None:
        record = {**MERITSHOT_RECORD, "perks": ["STIPEND"]}
        assert stipend_perk_consistency_check(record) is True


# ===========================================================================
# posting_burst_score
# ===========================================================================

# Build a realistic company batch: 5 postings, 3 within 7 days of target
BURST_COMPANY_RECORDS = [
    {**ANAKIN_RECORD, "datePublished": "2026-05-01"},   # target date
    {**ANAKIN_RECORD, "datePublished": "2026-05-03"},   # +2 days (in window)
    {**ANAKIN_RECORD, "datePublished": "2026-05-07"},   # +6 days (in window, boundary)
    {**ANAKIN_RECORD, "datePublished": "2026-05-09"},   # +8 days (outside window)
    {**ANAKIN_RECORD, "datePublished": "2026-04-01"},   # -30 days (outside window)
]
BURST_TARGET = {**ANAKIN_RECORD, "datePublished": "2026-05-01"}


class TestPostingBurstScore:

    def test_burst_count_correct(self) -> None:
        result = posting_burst_score(BURST_TARGET, BURST_COMPANY_RECORDS)
        # Target (May 1) + May 3 (+2) + May 7 (+6) = 3 in window
        assert result["burst_count"] == 3

    def test_no_company_records_returns_zero(self) -> None:
        result = posting_burst_score(BURST_TARGET, [])
        assert result["burst_count"] == 0
        assert result["cadence_days"] is None

    def test_missing_date_returns_zero(self) -> None:
        record = {**ANAKIN_RECORD}
        del record["datePublished"]
        result = posting_burst_score(record, BURST_COMPANY_RECORDS)
        assert result["burst_count"] == 0

    def test_cadence_days_computed(self) -> None:
        result = posting_burst_score(BURST_TARGET, BURST_COMPANY_RECORDS)
        # Sorted dates: Apr 1, May 1, May 3, May 7, May 9
        # Gaps: 30, 2, 4, 2 → mean = 9.5
        assert result["cadence_days"] == pytest.approx(9.5, rel=0.1)

    def test_single_record_no_cadence(self) -> None:
        result = posting_burst_score(BURST_TARGET, [BURST_TARGET])
        assert result["cadence_days"] is None

    def test_boundary_day_included(self) -> None:
        # A posting exactly 7 days away should be included (≤ window_days)
        records = [
            {**ANAKIN_RECORD, "datePublished": "2026-05-01"},
            {**ANAKIN_RECORD, "datePublished": "2026-05-08"},  # exactly +7
        ]
        result = posting_burst_score(
            {**ANAKIN_RECORD, "datePublished": "2026-05-01"},
            records,
        )
        assert result["burst_count"] == 2

    def test_custom_window_days(self) -> None:
        # With window=3, only +2 day posting is in-window (not +6)
        result = posting_burst_score(BURST_TARGET, BURST_COMPANY_RECORDS, window_days=3)
        assert result["burst_count"] == 2  # target + +2 day

    def test_mongo_date_format_parsed(self) -> None:
        records = [
            {**ANAKIN_RECORD, "datePublished": {"$date": "2026-05-01T00:00:00Z"}},
            {**ANAKIN_RECORD, "datePublished": {"$date": "2026-05-03T00:00:00Z"}},
        ]
        target = {**ANAKIN_RECORD, "datePublished": {"$date": "2026-05-01T00:00:00Z"}}
        result = posting_burst_score(target, records)
        assert result["burst_count"] == 2


# ===========================================================================
# deadline_urgency_score
# ===========================================================================

class TestDeadlineUrgencyScore:

    def test_missing_deadline_returns_none(self) -> None:
        record = {**MERITSHOT_RECORD}  # deadlineDate is None
        assert deadline_urgency_score(record) is None

    def test_missing_deadline_key_returns_none(self) -> None:
        record = {"name": "Intern", "company": "X", "datePublished": "2026-05-01"}
        assert deadline_urgency_score(record) is None

    def test_already_expired_returns_one(self) -> None:
        record = {**ANAKIN_RECORD, "deadlineDate": "2026-04-30", "datePublished": "2026-05-01"}
        assert deadline_urgency_score(record) == 1.0

    def test_same_day_deadline_returns_one(self) -> None:
        record = {**ANAKIN_RECORD, "deadlineDate": "2026-05-01", "datePublished": "2026-05-01"}
        assert deadline_urgency_score(record) == 1.0

    def test_very_short_gap_high_score(self) -> None:
        record = {**ANAKIN_RECORD, "deadlineDate": "2026-05-03", "datePublished": "2026-05-01"}
        score = deadline_urgency_score(record)  # 2 days
        assert score is not None and score > 0.8

    def test_short_gap_high_score(self) -> None:
        record = {**ANAKIN_RECORD, "deadlineDate": "2026-05-08", "datePublished": "2026-05-01"}
        score = deadline_urgency_score(record)  # 7 days
        assert score is not None and score > 0.6

    def test_medium_gap_moderate_score(self) -> None:
        record = {**ANAKIN_RECORD, "deadlineDate": "2026-05-20", "datePublished": "2026-05-01"}
        score = deadline_urgency_score(record)  # 19 days
        assert score is not None and 0.3 < score < 0.75

    def test_long_gap_low_score(self) -> None:
        record = {**ANAKIN_RECORD, "deadlineDate": "2026-08-01", "datePublished": "2026-05-01"}
        score = deadline_urgency_score(record)  # 92 days
        assert score is not None and score < 0.1

    def test_score_decreases_as_gap_increases(self) -> None:
        base = {"name": "Intern", "company": "X", "datePublished": "2026-05-01"}
        scores = []
        for d in [1, 7, 30, 90, 180]:
            dl = (date(2026, 5, 1) + timedelta(days=d)).isoformat()
            s = deadline_urgency_score({**base, "deadlineDate": dl})
            assert s is not None
            scores.append(s)
        # Each successive score should be <= the previous
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], f"scores[{i}]={scores[i]} not >= scores[{i+1}]={scores[i+1]}"

    def test_return_type_is_float_when_present(self) -> None:
        score = deadline_urgency_score(ANAKIN_RECORD)
        assert isinstance(score, float)

    def test_score_bounded(self) -> None:
        for record in [ANAKIN_RECORD, MOLEDRO_RECORD, ROOTSKY_RECORD]:
            score = deadline_urgency_score(record)
            if score is not None:
                assert 0.0 <= score <= 1.0

    def test_real_anakin_28_day_gap(self) -> None:
        # Anakin: published May 1, deadline May 29 → 28 days → moderate score
        score = deadline_urgency_score(ANAKIN_RECORD)
        assert score is not None and 0.3 < score < 0.7


# ===========================================================================
# openings_zscore
# ===========================================================================

OPENINGS_PEER_GROUP = [
    {**ANAKIN_RECORD,   "openings": 4},
    {**ROOTSKY_RECORD,  "openings": 2},
    {**EDITH_RECORD,    "openings": 2},
    {**MOLEDRO_RECORD,  "openings": 1},
    {**MERITSHOT_RECORD,"openings": 1},
]


class TestOpeningsZscore:

    def test_high_openings_positive_z(self) -> None:
        # 100 openings → far above mean → positive z
        record = {**ANAKIN_RECORD, "openings": 100}
        z = openings_zscore(record, OPENINGS_PEER_GROUP)
        assert z is not None and z > 0

    def test_low_openings_negative_z(self) -> None:
        # 1 opening → at low end → negative z
        record = {**ANAKIN_RECORD, "openings": 1}
        z = openings_zscore(record, OPENINGS_PEER_GROUP)
        assert z is not None and z < 0

    def test_none_openings_returns_none(self) -> None:
        record = {**ANAKIN_RECORD, "openings": None}
        assert openings_zscore(record, OPENINGS_PEER_GROUP) is None

    def test_missing_openings_key_returns_none(self) -> None:
        record = {"name": "Intern", "company": "X"}
        assert openings_zscore(record, OPENINGS_PEER_GROUP) is None

    def test_too_few_peers_returns_none(self) -> None:
        assert openings_zscore(ANAKIN_RECORD, [ROOTSKY_RECORD]) is None

    def test_empty_peer_group_returns_none(self) -> None:
        assert openings_zscore(ANAKIN_RECORD, []) is None

    def test_all_peers_same_value_returns_zero(self) -> None:
        peers = [{**ANAKIN_RECORD, "openings": 2}, {**ROOTSKY_RECORD, "openings": 2}]
        record = {**EDITH_RECORD, "openings": 2}
        assert openings_zscore(record, peers) == 0.0

    def test_return_type_is_float(self) -> None:
        z = openings_zscore(ANAKIN_RECORD, OPENINGS_PEER_GROUP)
        assert isinstance(z, float)

    def test_mean_value_scores_near_zero(self) -> None:
        # 2 openings is near the mean of [4,2,2,1,1] = 2.0
        record = {**ROOTSKY_RECORD, "openings": 2}
        z = openings_zscore(record, OPENINGS_PEER_GROUP)
        assert z is not None and abs(z) < 1.0


# ===========================================================================
# field_completeness_score
# ===========================================================================

class TestFieldCompletenessScore:

    def test_fully_complete_record_scores_one(self) -> None:
        # ANAKIN: all 8 checked fields populated
        score = field_completeness_score(ANAKIN_RECORD)
        assert score == 1.0

    def test_empty_record_scores_zero(self) -> None:
        assert field_completeness_score({}) == 0.0

    def test_none_record_scores_zero(self) -> None:
        assert field_completeness_score(None) == 0.0  # type: ignore[arg-type]

    def test_partial_record_scores_between(self) -> None:
        # Only skills and summary populated
        record = {"skills": ["Python"], "summary": "Great internship."}
        score = field_completeness_score(record)
        assert 0.0 < score < 1.0

    def test_meritshot_missing_skills_perks_lower_score(self) -> None:
        # Meritshot has no skills, no perks, no degree city exists
        meritshot_score = field_completeness_score(MERITSHOT_RECORD)
        anakin_score = field_completeness_score(ANAKIN_RECORD)
        assert meritshot_score < anakin_score

    def test_empty_lists_not_counted(self) -> None:
        record = {
            "skills": [],        # empty — doesn't count
            "field": ["CS"],     # populated — counts
            "responsibilities": [],
            "perks": [],
            "summary": "Hello.",
        }
        score = field_completeness_score(record)
        # 2/8 fields populated (field + summary)
        assert score == pytest.approx(2 / 8, rel=1e-3)

    def test_score_bounded(self) -> None:
        for record in [ANAKIN_RECORD, MOLEDRO_RECORD, MERITSHOT_RECORD, ROOTSKY_RECORD, {}]:
            if record is None:
                continue
            score = field_completeness_score(record)
            assert 0.0 <= score <= 1.0

    def test_return_type_is_float(self) -> None:
        assert isinstance(field_completeness_score(ANAKIN_RECORD), float)

    def test_whitespace_only_summary_not_counted(self) -> None:
        record = {**ANAKIN_RECORD, "summary": "   "}
        score_blank = field_completeness_score(record)
        score_full = field_completeness_score(ANAKIN_RECORD)
        assert score_blank < score_full

    def test_score_monotone_with_more_fields(self) -> None:
        # Adding fields one at a time should only increase the score
        prev = 0.0
        record: dict = {}
        additions = [
            ("skills", ["Python"]),
            ("field", ["Software Engineering"]),
            ("responsibilities", ["Build APIs"]),
            ("perks", ["Certificate"]),
            ("degree", ["B.Tech"]),
            ("tags", ["internship"]),
            ("summary", "Great internship opportunity."),
            ("city", "Bangalore"),
        ]
        for key, val in additions:
            record[key] = val
            score = field_completeness_score(record)
            assert score >= prev
            prev = score
