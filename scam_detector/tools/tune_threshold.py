"""
tune_threshold.py
-----------------
Threshold tuning utility for scam_detector.

Evaluates precision-recall curves across calibrated scam_score outputs on a labeled
dataset and finds optimal decision thresholds based on target precision, target recall,
or maximum F1 score.

Usage CLI:
    python -m scam_detector.tools.tune_threshold --dataset labeled_data.json [options]
    python tools/tune_threshold.py --dataset labeled_data.json --apply
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from scam_detector.config import Config, cfg as default_cfg
from scam_detector.feedback import _LABEL_MAP

log = logging.getLogger("scam_detector.tools.tune_threshold")


@dataclass
class ThresholdEvaluation:
    """Performance metrics evaluated at a single score threshold."""

    threshold: float
    precision: float
    recall: float
    f1_score: float
    tp: int
    fp: int
    tn: int
    fn: int


def compute_pr_curve(
    y_true: Sequence[int],
    y_scores: Sequence[float],
    step: float = 0.5,
) -> list[ThresholdEvaluation]:
    """
    Compute precision, recall, F1, and confusion matrix across thresholds in [0.0, 100.0].
    """
    if len(y_true) != len(y_scores):
        raise ValueError(f"Mismatch between y_true ({len(y_true)}) and y_scores ({len(y_scores)})")

    y_t = np.array(y_true, dtype=np.int32)
    y_s = np.array(y_scores, dtype=np.float32)

    total_positives = int(np.sum(y_t == 1))
    total_negatives = int(np.sum(y_t == 0))

    thresholds = [round(t, 2) for t in np.arange(0.0, 100.0 + step, step)]
    evaluations: list[ThresholdEvaluation] = []

    pos_mask = (y_t == 1)
    neg_mask = (y_t == 0)

    for th in thresholds:
        pred_pos = (y_s >= th)
        tp = int(np.sum(pred_pos & pos_mask))
        fp = int(np.sum(pred_pos & neg_mask))
        tn = total_negatives - fp
        fn = total_positives - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / total_positives if total_positives > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        evaluations.append(
            ThresholdEvaluation(
                threshold=th,
                precision=round(float(precision), 4),
                recall=round(float(recall), 4),
                f1_score=round(float(f1), 4),
                tp=tp,
                fp=fp,
                tn=tn,
                fn=fn,
            )
        )

    return evaluations


def find_optimal_threshold(
    curve: list[ThresholdEvaluation],
    target_precision: float | None = None,
    target_recall: float | None = None,
) -> ThresholdEvaluation:
    """
    Select optimal threshold from PR curve.

    - If target_precision is specified: lowest threshold achieving precision >= target_precision.
    - If target_recall is specified: threshold achieving recall >= target_recall (maximizing precision).
    - Otherwise (default): threshold maximizing F1 score.
    """
    if not curve:
        raise ValueError("Empty evaluation curve provided")

    if target_precision is not None:
        matching = [e for e in curve if e.precision >= target_precision]
        if matching:
            # Pick matching threshold with highest recall / lowest threshold
            return max(matching, key=lambda e: (e.recall, -e.threshold))
        log.warning("No threshold met target precision %.4f; returning highest precision", target_precision)
        return max(curve, key=lambda e: (e.precision, e.f1_score))

    if target_recall is not None:
        matching = [e for e in curve if e.recall >= target_recall]
        if matching:
            # Pick matching threshold with highest precision
            return max(matching, key=lambda e: (e.precision, e.f1_score))
        log.warning("No threshold met target recall %.4f; returning highest recall", target_recall)
        return max(curve, key=lambda e: (e.recall, e.f1_score))

    # Default: Maximize F1
    return max(curve, key=lambda e: (e.f1_score, e.precision))


def load_labeled_dataset(
    dataset_path: str | Path,
    config: Config | None = None,
) -> tuple[list[int], list[float]]:
    """
    Load a labeled dataset from JSON/JSONL/CSV file.
    Returns (y_true, y_scores) where y_true is 0/1 and y_scores is 0.0–100.0.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    records: list[dict[str, Any]] = []
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        label_col = next((c for c in ("is_fake_posting", "label", "is_scam") if c in df.columns), None)
        score_col = next((c for c in ("fraud_score", "scam_score") if c in df.columns), None)
        if label_col and score_col:
            y_t = df[label_col].astype(int).tolist()
            y_s = df[score_col].astype(float).tolist()
            return y_t, y_s
        records = df.to_dict(orient="records")
    elif path.suffix in (".jsonl", ".ndjson"):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    else:
        with path.open(encoding="utf-8") as fh:
            payload = json.load(fh)
            if isinstance(payload, list):
                records = payload
            elif isinstance(payload, dict):
                records = payload.get("records") or payload.get("items") or payload.get("data") or [payload]

    if not records:
        raise ValueError(f"No records found in {path}")

    y_true: list[int] = []
    y_scores: list[float] = []

    need_scoring_records: list[dict[str, Any]] = []
    need_scoring_indices: list[int] = []

    for i, r in enumerate(records):
        # Extract label
        label: int | None = None
        if "label" in r and r["label"] is not None:
            label = int(r["label"])
        elif "is_scam" in r and r["is_scam"] is not None:
            label = 1 if bool(r["is_scam"]) else 0
        elif "is_fake_posting" in r and r["is_fake_posting"] is not None:
            label = int(r["is_fake_posting"])
        elif "reviewer_decision" in r and r["reviewer_decision"] in _LABEL_MAP:
            label = _LABEL_MAP[r["reviewer_decision"]]

        if label is None:
            log.warning("Skipping record at index %d due to missing label", i)
            continue

        y_true.append(label)

        # Extract score if precomputed
        if "scam_score" in r and r["scam_score"] is not None:
            score = float(r["scam_score"])
            if score <= 1.0 and score > 0.0:
                score *= 100.0
            y_scores.append(score)
        elif "fraud_score" in r and r["fraud_score"] is not None:
            score = float(r["fraud_score"])
            if score <= 1.0 and score > 0.0:
                score *= 100.0
            y_scores.append(score)
        else:
            # Map Kaggle columns if needed for raw pipeline scoring
            mapped_r = r
            if "internship_title" in r or "company_name" in r:
                summary_parts = []
                if r.get("payment_required") == 1 or (r.get("registration_fee") or 0) > 0:
                    summary_parts.append(f"Upfront registration fee required: ₹{r.get('registration_fee', 0)}. Security deposit payment mandated.")
                if r.get("fake_certificate_offer") == 1:
                    summary_parts.append("Guaranteed certificate offer upon fee payment.")
                if r.get("vague_description_score", 0) > 40:
                    summary_parts.append("Generic online work from home role with vague details.")
                if r.get("phishing_language_score", 0) > 30:
                    summary_parts.append("Urgent hiring! Send bank details and ID proof immediately via WhatsApp.")

                summary = " ".join(summary_parts) if summary_parts else f"Internship opportunity for {r.get('internship_title', 'Role')} in {r.get('industry', 'Industry')}."

                mapped_r = {
                    "_id": f"{r.get('posting_date', '')}_{r.get('company_name', '')}_{i}",
                    "name": r.get("internship_title", "Intern"),
                    "company": r.get("company_name", "Company"),
                    "datePublished": r.get("posting_date", ""),
                    "city": r.get("location", ""),
                    "isRemote": (str(r.get("work_mode", "")).lower() == "remote"),
                    "stipend": r.get("stipend"),
                    "summary": summary,
                    "responsibilities": summary,
                    "applyLink": f"https://{str(r.get('company_name', '')).lower().replace(' ', '')}.com/jobs" if r.get("website_available") == 1 else "http://free-email-apply-form.biz",
                    "source": "kaggle_internship",
                }
            need_scoring_records.append(mapped_r)
            need_scoring_indices.append(len(y_true) - 1)
            y_scores.append(0.0)

    if need_scoring_records:
        log.info("Computing scam_scores for %d raw records via pipeline...", len(need_scoring_records))
        from scam_detector.pipeline import process_records

        scored = process_records(need_scoring_records, config=config)
        for idx, s_rec in zip(need_scoring_indices, scored):
            y_scores[idx] = float(s_rec.get("scam_score", 0.0))

    return y_true, y_scores


def apply_thresholds_to_config(
    block_threshold: float,
    review_threshold: float | None = None,
    config_path: str | Path | None = None,
) -> None:
    """
    Update decision threshold defaults in config.py.
    """
    path = Path(config_path) if config_path else Path("scam_detector/config.py")
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path}")

    content = path.read_text(encoding="utf-8")

    # Update DecisionThresholds clear_below and block_at_or_above defaults
    if review_threshold is not None:
        content = re.sub(
            r"(class DecisionThresholds\(BaseModel\):.*?\bclear_below:\s*float\s*=\s*Field\(\s*default=)[\d\.]+",
            rf"\g<1>{review_threshold:.1f}",
            content,
            flags=re.DOTALL,
        )

    content = re.sub(
        r"(class DecisionThresholds\(BaseModel\):.*?\bblock_at_or_above:\s*float\s*=\s*Field\(\s*default=)[\d\.]+",
        rf"\g<1>{block_threshold:.1f}",
        content,
        flags=re.DOTALL,
    )

    path.write_text(content, encoding="utf-8")
    log.info("Updated config.py defaults: block_at_or_above=%.1f", block_threshold)
    if review_threshold is not None:
        log.info("Updated config.py defaults: clear_below=%.1f", review_threshold)


def generate_tuning_report(
    eval_res: ThresholdEvaluation,
    total_samples: int,
    target_prec: float | None = None,
    target_rec: float | None = None,
) -> str:
    """Generate human-readable evaluation report."""
    lines = [
        "============================================================",
        "  SCAM DETECTOR — THRESHOLD TUNING REPORT",
        "============================================================",
        f"  Total Labeled Samples : {total_samples}",
        f"  Target Mode           : "
        + (
            f"Precision >= {target_prec:.4f}"
            if target_prec is not None
            else f"Recall >= {target_rec:.4f}"
            if target_rec is not None
            else "Maximize F1 Score"
        ),
        "------------------------------------------------------------",
        f"  Recommended Block Threshold : {eval_res.threshold:.2f} / 100",
        f"  Precision                   : {eval_res.precision:.4f} ({eval_res.precision * 100:.2f}%)",
        f"  Recall                      : {eval_res.recall:.4f} ({eval_res.recall * 100:.2f}%)",
        f"  F1 Score                    : {eval_res.f1_score:.4f}",
        "------------------------------------------------------------",
        "  Confusion Matrix:",
        f"    True Positives (TP)  : {eval_res.tp}",
        f"    False Positives (FP) : {eval_res.fp}",
        f"    True Negatives (TN)  : {eval_res.tn}",
        f"    False Negatives (FN) : {eval_res.fn}",
        "============================================================",
    ]
    return "\n".join(lines)


def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Tune scam_score decision thresholds on a labeled dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Path to labeled dataset (JSON, JSONL, or CSV)")
    parser.add_argument("--target-precision", type=float, default=None, help="Target precision threshold (0.0–1.0)")
    parser.add_argument("--target-recall", type=float, default=None, help="Target recall threshold (0.0–1.0)")
    parser.add_argument("--review-threshold", type=float, default=None, help="Optional explicit review threshold")
    parser.add_argument("--apply", action="store_true", help="Automatically update config.py with tuned thresholds")
    parser.add_argument("--config-path", type=str, default=None, help="Path to config.py to update when --apply is set")

    args = parser.parse_args()

    y_true, y_scores = load_labeled_dataset(args.dataset)
    curve = compute_pr_curve(y_true, y_scores)
    optimal = find_optimal_threshold(curve, target_precision=args.target_precision, target_recall=args.target_recall)

    report = generate_tuning_report(
        optimal,
        total_samples=len(y_true),
        target_prec=args.target_precision,
        target_rec=args.target_recall,
    )
    print(report)

    if args.apply:
        apply_thresholds_to_config(
            block_threshold=optimal.threshold,
            review_threshold=args.review_threshold,
            config_path=args.config_path,
        )
        print("Config updated successfully.")


if __name__ == "__main__":
    main()
