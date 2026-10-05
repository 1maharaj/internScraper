"""
test_ask5_features.py
---------------------
Tests for ASK 5 upgraded features:
- WHOIS domain_age_days with SQLite caching & failure handling (None)
- scam_corpus_similarity using sentence-transformers & FeedbackStore
- recruiter_posting_velocity_24h and _72h
- Peer group conditioning on (role_category, city_tier, company_size_tier)
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest
import numpy as np

from scam_detector.features.company_features import (
    CompanyFeatures,
    fetch_domain_age_days,
    extract_company_url_features,
)
from scam_detector.features.text_features import (
    TextFeatureVector,
    scam_corpus_similarity,
    get_scam_corpus_embeddings,
    extract_text_features,
)
from scam_detector.features.temporal_features import (
    TemporalFeatures,
    recruiter_posting_velocity,
)
from scam_detector.features.stipend_features import (
    get_role_category,
    get_city_tier,
    get_company_size_tier,
)
from scam_detector.pipeline import build_peer_group, process_records
from scam_detector.feedback import FeedbackStore, ReviewFeedback


# ===========================================================================
# 1. WHOIS domain_age_days Tests
# ===========================================================================

def test_whois_domain_age_failure_returns_none(tmp_path):
    cache_db = tmp_path / "test_whois.sqlite"
    # Unparseable / non-existent domain should gracefully return None, not 0
    age = fetch_domain_age_days("nonexistent-domain-12345-xyz.invalid", cache_db_path=cache_db)
    assert age is None


def test_whois_domain_age_sqlite_caching(tmp_path):
    cache_db = tmp_path / "test_whois.sqlite"
    # Manually populate cache to test sqlite retrieval
    import sqlite3
    import time
    conn = sqlite3.connect(str(cache_db))
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS whois_cache (
                domain TEXT PRIMARY KEY,
                age_days INTEGER,
                fetched_at REAL
            )
            """
        )
        conn.execute(
            "INSERT INTO whois_cache VALUES (?, ?, ?)",
            ("example.com", 1500, time.time()),
        )
    conn.close()

    age = fetch_domain_age_days("example.com", cache_db_path=cache_db)
    assert age == 1500


def test_company_features_has_domain_age_days_field():
    cf = CompanyFeatures(domain_age_days=100)
    assert cf.domain_age_days == 100
    cf_none = CompanyFeatures(domain_age_days=None)
    assert cf_none.domain_age_days is None


# ===========================================================================
# 2. scam_corpus_similarity Tests
# ===========================================================================

def test_scam_corpus_similarity_empty_embeddings():
    sim = scam_corpus_similarity("Urgent hiring pay fee now", None)
    assert sim == 0.0
    sim_empty = scam_corpus_similarity("Urgent hiring pay fee now", np.empty((0, 384)))
    assert sim_empty == 0.0


def test_scam_corpus_similarity_mock_embeddings():
    # 1D or dummy 2D array
    dummy_scam_embs = np.random.randn(3, 384).astype(np.float32)
    # L2 normalise
    norms = np.linalg.norm(dummy_scam_embs, axis=1, keepdims=True)
    dummy_scam_embs = dummy_scam_embs / norms

    sim = scam_corpus_similarity("Backend developer intern", dummy_scam_embs)
    assert 0.0 <= sim <= 1.0


def test_get_scam_corpus_embeddings_with_feedback(tmp_path):
    fb_path = tmp_path / "feedback.jsonl"
    store = FeedbackStore(fb_path)
    store.record_feedback(ReviewFeedback(record_id="scam-1", reviewer_decision="confirmed_scam"))
    store.record_feedback(ReviewFeedback(record_id="legit-1", reviewer_decision="confirmed_legit"))

    records = [
        {"_id": "scam-1", "name": "Urgent Fee Job", "summary": "Pay deposit first"},
        {"_id": "legit-1", "name": "Software Intern", "summary": "Good company"},
    ]

    embs = get_scam_corpus_embeddings(store, records)
    if embs is not None:
        assert embs.shape[0] == 1  # 1 confirmed scam record


# ===========================================================================
# 3. recruiter_posting_velocity Tests
# ===========================================================================

def test_recruiter_posting_velocity_24h_and_72h():
    now_utc = datetime.now(timezone.utc)
    t0 = now_utc.isoformat()
    t_12h = (now_utc - timedelta(hours=12)).isoformat()
    t_48h = (now_utc - timedelta(hours=48)).isoformat()
    t_100h = (now_utc - timedelta(hours=100)).isoformat()

    all_records = [
        {"_id": "rec-1", "recruiter_id": "user_42", "datePublished": t0},
        {"_id": "rec-2", "recruiter_id": "user_42", "datePublished": t_12h},
        {"_id": "rec-3", "recruiter_id": "user_42", "datePublished": t_48h},
        {"_id": "rec-4", "recruiter_id": "user_42", "datePublished": t_100h},
        {"_id": "rec-5", "recruiter_id": "user_99", "datePublished": t0},
    ]

    target = all_records[0]
    v24 = recruiter_posting_velocity(target, all_records, hours=24)
    v72 = recruiter_posting_velocity(target, all_records, hours=72)

    # Within 24h of t0: t0 (0h) and t_12h (12h) -> 2 postings
    assert v24 == 2
    # Within 72h of t0: t0, t_12h, t_48h -> 3 postings
    assert v72 == 3


# ===========================================================================
# 4. Peer Group Conditioning Tests
# ===========================================================================

def test_peer_group_category_extraction():
    r1 = {
        "name": "Backend Python Developer",
        "city_tier": "tier_1",
        "company_size": 25,
    }
    assert get_role_category(r1) == "tech"
    assert get_city_tier(r1) == "tier_1"
    assert get_company_size_tier(r1) == "startup"

    r2 = {
        "name": "Social Media Executive",
        "isRemote": True,
        "company_size_tier": "enterprise",
    }
    assert get_role_category(r2) == "marketing_sales"
    assert get_city_tier(r2) == "remote"
    assert get_company_size_tier(r2) == "enterprise"


def test_build_peer_group_conditioning():
    all_recs = [
        {"_id": "1", "name": "Backend Dev", "city_tier": "tier_1", "company_size": 20},
        {"_id": "2", "name": "Frontend Dev", "city_tier": "tier_1", "company_size": 30},
        {"_id": "3", "name": "Marketing Exec", "city_tier": "tier_1", "company_size": 20},
        {"_id": "4", "name": "Backend Dev", "city_tier": "tier_2", "company_size": 500},
    ]

    # Searching peers for record "1" (tech, tier_1, startup)
    # Record "2" is also (tech, tier_1, startup) -> peers should be [rec 1, rec 2]
    peers = build_peer_group(all_recs[0], all_recs)
    peer_ids = [r["_id"] for r in peers]
    assert "1" in peer_ids
    assert "2" in peer_ids
    assert "3" not in peer_ids  # different role category
