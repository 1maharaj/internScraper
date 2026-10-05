"""
test_supervised_model.py
------------------------
Unit and integration tests for scam_detector.scoring.supervised_model.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest

from scam_detector.config import Config
from scam_detector.feedback import ReviewFeedback
from scam_detector.features import extract_all
from scam_detector.data_quality.remediate import remediate_record
from scam_detector.scoring.supervised_model import (
    SupervisedScamModel,
    SupervisedResult,
    MODEL_VERSION,
)
from scam_detector.scoring.risk_engine import RiskEngine
from scam_detector.scoring.rules_engine import RulesEngine, RuleInput
from scam_detector.pipeline import process_records


def _make_dummy_record(idx: int, is_scam: bool = False) -> dict:
    if is_scam:
        return {
            "_id": f"rec-{idx}",
            "name": "URGENT PAYMENT NEEDED FOR INTERNSHIP",
            "company": "Digital Marketing",
            "summary": "Pay Rs 5000 registration fee to get this internship. Send bank details.",
            "stipend": "Unpaid",
            "perks": ["Stipend"],
            "applyLink": "http://bit.ly/fake-link",
            "isRemote": True,
        }
    return {
        "_id": f"rec-{idx}",
        "name": "Software Engineering Intern",
        "company": "Google LLC",
        "summary": "Join our engineering team to build scalable systems in Python and C++.",
        "stipend": {"type": "fixed", "amount": 25000, "currency": "INR", "period": "monthly"},
        "perks": ["Certificate", "Letter of recommendation"],
        "applyLink": "https://careers.google.com/jobs/123",
        "isRemote": False,
        "skills": ["Python", "C++"],
        "field": ["Engineering"],
    }


def test_unfitted_model_graceful_noop(tmp_path: Path):
    model_file = tmp_path / "non_existent_model.joblib"
    model = SupervisedScamModel(model_path=model_file)

    assert not model.is_fitted
    assert model.load() is False

    # Predict proba on empty frame or invalid frame returns empty list
    import pandas as pd
    assert model.predict_proba(pd.DataFrame()) == []

    res = model.score_record(extract_all(_make_dummy_record(1)))
    assert isinstance(res, SupervisedResult)
    assert res.model_available is False
    assert res.scam_probability is None


def test_fit_insufficient_samples_raises_error(tmp_path: Path):
    model_file = tmp_path / "model.joblib"
    model = SupervisedScamModel(model_path=model_file, min_training_samples=100)

    records = [_make_dummy_record(i) for i in range(10)]
    fvs = [extract_all(r) for r in records]
    feedback = [
        ReviewFeedback(record_id=f"rec-{i}", reviewer_decision="confirmed_legit")
        for i in range(10)
    ]

    with pytest.raises(ValueError, match="Insufficient labeled samples"):
        model.fit(records, fvs, feedback, min_samples=100)


def test_fit_save_load_predict(tmp_path: Path):
    model_file = tmp_path / "supervised_model.joblib"
    model = SupervisedScamModel(model_path=model_file, min_training_samples=50)

    # Generate 60 synthetic labeled records (30 scam, 30 legit)
    records = []
    fvs = []
    feedback = []
    for i in range(60):
        is_scam = (i % 2 == 0)
        rec = _make_dummy_record(i, is_scam=is_scam)
        records.append(rec)
        fvs.append(extract_all(rec))
        decision = "confirmed_scam" if is_scam else "confirmed_legit"
        feedback.append(ReviewFeedback(record_id=f"rec-{i}", reviewer_decision=decision))

    # Fit model with min_samples=50
    fitted_model = model.fit(records, fvs, feedback, min_samples=50)
    assert fitted_model.is_fitted
    assert model_file.exists()
    assert fitted_model.model_version == MODEL_VERSION
    assert len(fitted_model.feature_columns) > 0

    # Test loading in a fresh instance
    new_model = SupervisedScamModel(model_path=model_file)
    assert new_model.load() is True
    assert new_model.is_fitted

    # Test single-record scoring
    scam_rec = _make_dummy_record(999, is_scam=True)
    scam_fv = extract_all(scam_rec)
    res_scam = new_model.score_record(scam_fv, record=scam_rec)
    assert res_scam.model_available is True
    assert isinstance(res_scam.scam_probability, float)
    assert 0.0 <= res_scam.scam_probability <= 1.0

    legit_rec = _make_dummy_record(1000, is_scam=False)
    legit_fv = extract_all(legit_rec)
    res_legit = new_model.score_record(legit_fv, record=legit_rec)
    assert res_legit.model_available is True
    assert isinstance(res_legit.scam_probability, float)


def test_risk_engine_blend_with_supervised_score():
    cfg = Config()
    cfg.flags.enable_ml_risk_engine = True
    cfg.blend_weights.rules_weight = 0.40
    cfg.blend_weights.anomaly_weight = 0.20
    cfg.blend_weights.supervised_weight = 0.40

    risk_engine = RiskEngine(config=cfg)
    rules_engine = RulesEngine(config=cfg)

    rec = _make_dummy_record(1, is_scam=False)
    rem = remediate_record(rec)
    rule_input = RuleInput()
    rule_res = rules_engine.run(rule_input)

    # Test with supervised_score=0.90
    result = risk_engine.score_record(
        record=rem,
        rule_result=rule_res,
        anomaly_score=0.10,
        confidence_score=1.0,
        supervised_score=0.90,
    )

    assert result.scam_score > 30.0  # supervised score raised the blended score


def test_pipeline_with_supervised_model(tmp_path: Path):
    # Setup trained model
    model_file = tmp_path / "supervised_model.joblib"
    model = SupervisedScamModel(model_path=model_file, min_training_samples=20)

    records = []
    fvs = []
    feedback = []
    for i in range(30):
        is_scam = (i % 2 == 0)
        rec = _make_dummy_record(i, is_scam=is_scam)
        records.append(rec)
        fvs.append(extract_all(rec))
        decision = "confirmed_scam" if is_scam else "confirmed_legit"
        feedback.append(ReviewFeedback(record_id=f"rec-{i}", reviewer_decision=decision))

    model.fit(records, fvs, feedback, min_samples=20)

    # Configure custom pipeline config to use trained model
    cfg = Config()
    cfg.supervised.model_path = str(model_file)
    cfg.flags.enable_ml_risk_engine = True

    test_input = [_make_dummy_record(101, is_scam=False), _make_dummy_record(102, is_scam=True)]
    processed = process_records(test_input, config=cfg)

    assert len(processed) == 2
    assert "scam_score" in processed[0]
    assert "decision" in processed[0]
