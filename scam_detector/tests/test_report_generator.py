"""
test_report_generator.py
-------------------------
Tests for Markdown and HTML audit report generation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from scam_detector.tools.report_generator import (
    generate_markdown_audit_report,
    generate_html_audit_report,
)


def test_generate_markdown_audit_report() -> None:
    scored_records = [
        {
            "_id": "rec1",
            "company": "Fake Corp",
            "name": "Data Entry Intern",
            "scam_score": 85.0,
            "decision": "block",
            "confidence": 0.95,
            "explanation_summary": "Sensitive info requested (bank account details)",
        },
        {
            "_id": "rec2",
            "company": "Good Tech",
            "name": "Frontend Intern",
            "scam_score": 15.0,
            "decision": "clear",
            "confidence": 0.90,
            "explanation_summary": "No risk triggers",
        },
    ]

    graph_metrics = {
        "total_nodes": 10,
        "company_count": 8,
        "domain_count": 2,
        "connected_components_count": 5,
        "multi_company_networks_count": 1,
        "largest_component_size": 4,
    }

    md = generate_markdown_audit_report("Sample Test Corpus", scored_records, graph_metrics)
    assert "# Scam Detection Audit Report: Sample Test Corpus" in md
    assert "Blocked (High Risk): 1" in md
    assert "Coordinated Multi-Company Networks" in md
    assert "| Fake Corp |" in md


def test_generate_html_audit_report() -> None:
    scored_records = [
        {
            "_id": "rec1",
            "company": "Fake Corp",
            "name": "Data Entry Intern",
            "scam_score": 85.0,
            "decision": "block",
            "confidence": 0.95,
            "explanation_summary": "Sensitive info requested",
            "shared_infrastructure": True,
            "duplicate_cluster_network_size": 5,
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "audit_report.html"
        html = generate_html_audit_report(
            "Test HTML Corpus",
            scored_records,
            output_path=out_path,
        )

        assert "<title>Scam Detector Audit Report - Test HTML Corpus</title>" in html
        assert "Fake Corp" in html
        assert "badge-block" in html
        assert out_path.exists()
        assert len(out_path.read_text(encoding="utf-8")) > 500
