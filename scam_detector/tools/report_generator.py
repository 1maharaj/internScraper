"""
report_generator.py
--------------------
Generate interactive HTML and Markdown audit and review reports for scored datasets.
Embeds score distributions, risk badges, SHAP feature attributions, and coordinated network graphs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


def generate_markdown_audit_report(
    dataset_name: str,
    scored_records: list[dict[str, Any]],
    graph_metrics: dict[str, Any] | None = None,
) -> str:
    """
    Generate a clean GitHub-flavored Markdown report of the scored corpus.
    """
    total = len(scored_records)
    blocks = sum(1 for r in scored_records if r.get("decision") == "block")
    reviews = sum(1 for r in scored_records if r.get("decision") == "review")
    clears = sum(1 for r in scored_records if r.get("decision") == "clear")

    avg_score = sum(float(r.get("scam_score", 0.0)) for r in scored_records) / max(1, total)

    lines: list[str] = [
        f"# Scam Detection Audit Report: {dataset_name}",
        "",
        "## Executive Summary",
        "",
        f"- **Total Postings Analyzed**: {total:,}",
        f"- **Average Scam Score**: {avg_score:.2f} / 100",
        f"- 🚨 **Blocked (High Risk): {blocks:,}** ({blocks/max(1, total):.1%})",
        f"- ⚠️ **Flagged for Review (Medium Risk): {reviews:,}** ({reviews/max(1, total):.1%})",
        f"- ✅ **Auto-Approved / Clear (Low Risk): {clears:,}** ({clears/max(1, total):.1%})",
        "",
    ]

    if graph_metrics:
        lines.extend([
            "## Infrastructure & Organization Graph Insights",
            "",
            f"- **Total Nodes**: {graph_metrics.get('total_nodes', 0):,} ({graph_metrics.get('company_count', 0):,} companies, {graph_metrics.get('domain_count', 0):,} domains)",
            f"- **Connected Components**: {graph_metrics.get('connected_components_count', 0):,}",
            f"- **Coordinated Multi-Company Networks (≥3 companies)**: {graph_metrics.get('multi_company_networks_count', 0):,}",
            f"- **Largest Network Scale**: {graph_metrics.get('largest_component_size', 0):,} entities",
            "",
        ])

    lines.extend([
        "## Top High-Risk Flagged Postings",
        "",
        "| ID | Company | Role | Scam Score | Decision | Confidence | Flag Summary |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    sorted_records = sorted(scored_records, key=lambda x: float(x.get("scam_score", 0.0)), reverse=True)
    for r in sorted_records[:15]:
        rid = str(r.get("_id") or r.get("id") or "N/A")[:12]
        co = str(r.get("company") or "Unknown")[:25]
        title = str(r.get("name") or r.get("title") or "Role")[:30]
        score = float(r.get("scam_score", 0.0))
        dec = str(r.get("decision", "clear")).upper()
        conf = f"{float(r.get('confidence', 1.0)):.2f}"
        summ = str(r.get("explanation_summary", "N/A"))[:50]
        lines.append(f"| {rid} | {co} | {title} | **{score:.1f}** | `{dec}` | {conf} | {summ} |")

    lines.append("")
    return "\n".join(lines)


def generate_html_audit_report(
    dataset_name: str,
    scored_records: list[dict[str, Any]],
    graph_metrics: dict[str, Any] | None = None,
    output_path: str | Path | None = None,
) -> str:
    """
    Generate a styled, stand-alone HTML report with responsive design and risk badges.
    """
    total = len(scored_records)
    blocks = sum(1 for r in scored_records if r.get("decision") == "block")
    reviews = sum(1 for r in scored_records if r.get("decision") == "review")
    clears = sum(1 for r in scored_records if r.get("decision") == "clear")
    avg_score = sum(float(r.get("scam_score", 0.0)) for r in scored_records) / max(1, total)

    sorted_records = sorted(scored_records, key=lambda x: float(x.get("scam_score", 0.0)), reverse=True)

    rows_html = []
    for i, r in enumerate(sorted_records[:50], 1):
        score = float(r.get("scam_score", 0.0))
        dec = str(r.get("decision", "clear")).lower()
        badge_cls = "badge-block" if dec == "block" else ("badge-review" if dec == "review" else "badge-clear")
        co = str(r.get("company") or "Unknown")
        title = str(r.get("name") or r.get("title") or "Role")
        summ = str(r.get("explanation_summary", "(no triggers)"))
        conf = float(r.get("confidence", 1.0))
        shared = "🚨 Yes" if r.get("shared_infrastructure") else "No"
        net_size = int(r.get("duplicate_cluster_network_size", 1))

        rows_html.append(f"""
        <tr>
            <td>{i}</td>
            <td><strong>{co}</strong></td>
            <td>{title}</td>
            <td><span class="score">{score:.1f}</span></td>
            <td><span class="badge {badge_cls}">{dec.upper()}</span></td>
            <td>{conf:.2f}</td>
            <td>{shared} (size {net_size})</td>
            <td class="summary-cell">{summ}</td>
        </tr>
        """)

    graph_html = ""
    if graph_metrics:
        graph_html = f"""
        <div class="card">
            <h3>Graph & Infrastructure Network Intelligence</h3>
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-value">{graph_metrics.get('total_nodes', 0):,}</div>
                    <div class="stat-label">Total Graph Entities</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{graph_metrics.get('multi_company_networks_count', 0):,}</div>
                    <div class="stat-label">Multi-Company Networks (≥3)</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{graph_metrics.get('largest_component_size', 0):,}</div>
                    <div class="stat-label">Largest Network Scale</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{graph_metrics.get('density', 0.0):.4f}</div>
                    <div class="stat-label">Graph Density</div>
                </div>
            </div>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Scam Detector Audit Report - {dataset_name}</title>
    <style>
        :root {{
            --bg: #0f172a;
            --card-bg: #1e293b;
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --border: #334155;
            --accent: #38bdf8;
            --danger: #ef4444;
            --warning: #f59e0b;
            --success: #10b981;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 32px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1, h2, h3 {{
            color: #ffffff;
            font-weight: 700;
        }}
        .header {{
            margin-bottom: 32px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 16px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .stat-box {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2rem;
            font-weight: 800;
            color: var(--accent);
        }}
        .stat-value.danger {{ color: var(--danger); }}
        .stat-value.warning {{ color: var(--warning); }}
        .stat-value.success {{ color: var(--success); }}
        .stat-label {{
            color: var(--text-muted);
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 4px;
        }}
        .card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 24px;
            margin-bottom: 32px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}
        th, td {{
            padding: 12px 14px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            background: #0f172a;
            color: var(--text-muted);
            font-weight: 600;
        }}
        tr:hover td {{
            background: #27354a;
        }}
        .score {{
            font-weight: 700;
            color: var(--accent);
        }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.05em;
        }}
        .badge-block {{ background: rgba(239, 68, 68, 0.2); color: var(--danger); }}
        .badge-review {{ background: rgba(245, 158, 11, 0.2); color: var(--warning); }}
        .badge-clear {{ background: rgba(16, 185, 129, 0.2); color: var(--success); }}
        .summary-cell {{
            color: var(--text-muted);
            max-width: 320px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ Scam Detection Audit Report</h1>
            <p style="color: var(--text-muted);">Dataset: <strong>{dataset_name}</strong> | Generated by iFind AI Scam Engine</p>
        </div>

        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-value">{total:,}</div>
                <div class="stat-label">Total Listings</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{avg_score:.1f}</div>
                <div class="stat-label">Average Scam Score</div>
            </div>
            <div class="stat-box">
                <div class="stat-value danger">{blocks:,}</div>
                <div class="stat-label">Blocked (High Risk)</div>
            </div>
            <div class="stat-box">
                <div class="stat-value warning">{reviews:,}</div>
                <div class="stat-label">Review Required</div>
            </div>
            <div class="stat-box">
                <div class="stat-value success">{clears:,}</div>
                <div class="stat-label">Auto-Cleared</div>
            </div>
        </div>

        {graph_html}

        <div class="card">
            <h3>Top Risk Ranked Postings</h3>
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Company</th>
                        <th>Role Title</th>
                        <th>Score</th>
                        <th>Decision</th>
                        <th>Confidence</th>
                        <th>Network</th>
                        <th>Verdict & Primary Signals</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(rows_html)}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>"""

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(html, encoding="utf-8")

    return html
