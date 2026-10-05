"""
run_graph_analysis.py
---------------------
Run graph-based analysis on the data sources (Internshala and Unstop)
to detect coordinate posting networks and export visualizations/GraphML.
"""

from __future__ import annotations

import json
import os
import sys
import logging
from pathlib import Path
import networkx as nx

# Ensure stdout handles UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure scam_detector is on sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from scam_detector.features.graph_features import (
    build_company_infrastructure_graph,
    export_largest_components_visualization,
)
from scam_detector.pipeline import _build_duplicate_neighbors

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("run_graph_analysis")

ARTIFACT_DIR = Path(
    os.environ.get(
        "ARTIFACT_DIR",
        "C:/Users/vedant patil/.gemini/antigravity-ide/brain/fa7b41d5-181b-41da-adda-c5cd2be2b0df",
    )
)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def analyze_dataset(name: str, records: list[dict]) -> None:
    print("\n" + "=" * 70)
    print(f" ANALYZING DATASET: {name} ({len(records)} records)")
    print("=" * 70)

    # 1. Build duplicate neighbors index
    print("Building DuplicateIndex / finding duplicate neighbors...")
    neighbors_by_id = _build_duplicate_neighbors(records)

    # 2. Build graph
    print("Building company-infrastructure graph...")
    graph = build_company_infrastructure_graph(records, neighbors_by_id)

    # 3. Graph metrics
    companies = [n for n in graph.nodes if n.startswith("company:")]
    domains = [n for n in graph.nodes if n.startswith("domain:")]
    print(f"Graph stats: {graph.number_of_nodes()} total nodes ({len(companies)} companies, {len(domains)} domains), {graph.number_of_edges()} edges")

    # Connected components
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    print(f"Total connected components: {len(components)}")
    
    # Analyze components size distribution by company count
    size_dist: dict[int, int] = {}
    multi_company_components = []

    for comp in components:
        comp_companies = [n for n in comp if n.startswith("company:")]
        co_count = len(comp_companies)
        size_dist[co_count] = size_dist.get(co_count, 0) + 1
        if co_count >= 3:
            multi_company_components.append(comp)

    print("\nConnected Component Size Distribution (by company count):")
    for size in sorted(size_dist.keys()):
        print(f"  - Components with {size} companies: {size_dist[size]}")

    # Check for clusters of 3+ companies
    if multi_company_components:
        print(f"\n[ALERT] Found {len(multi_company_components)} coordinated network(s) of 3+ companies!")
        for idx, comp in enumerate(multi_company_components, 1):
            comp_companies = [comp_nodes.split(":", 1)[1] for comp_nodes in comp if comp_nodes.startswith("company:")]
            comp_domains = [comp_nodes.split(":", 1)[1] for comp_nodes in comp if comp_nodes.startswith("domain:")]
            print(f"  Network #{idx}:")
            print(f"    - Companies ({len(comp_companies)}): {', '.join(comp_companies)}")
            if comp_domains:
                print(f"    - Shared Domains ({len(comp_domains)}): {', '.join(comp_domains)}")
            else:
                print("    - Shared Domains: None (linked purely via near-duplicate text)")
    else:
        print("\nNo coordinated networks of 3+ companies found in this dataset.")

    # 4. Export visualization and GraphML
    img_name = f"{name.lower().replace(' ', '_')}_graph.png"
    graphml_name = f"{name.lower().replace(' ', '_')}_graph.graphml"
    
    img_path = ARTIFACT_DIR / img_name
    graphml_path = ARTIFACT_DIR / graphml_name

    print(f"\nExporting visualization to {img_path}...")
    export_largest_components_visualization(
        graph,
        output_image_path=str(img_path),
        output_graphml_path=str(graphml_path),
        top_n=3
    )
    print(f"Exported GraphML to {graphml_path}")


def main() -> None:
    # ── Source 1: Internshala ──
    ishala_path = root_dir / "web scrapper" / "intershala_scraper" / "internships.json"
    if ishala_path.exists():
        with open(ishala_path, encoding="utf-8") as f:
            records = json.load(f)
        analyze_dataset("Internshala", records)
    else:
        print(f"Internshala dataset not found at {ishala_path}")

    # ── Source 2: Unstop ──
    unstop_path = root_dir / "unstop_scam_scored.json"
    if unstop_path.exists():
        with open(unstop_path, encoding="utf-8") as f:
            records = json.load(f)
        analyze_dataset("Unstop", records)
    else:
        print(f"Unstop dataset not found at {unstop_path}")


if __name__ == "__main__":
    main()
