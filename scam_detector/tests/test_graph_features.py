"""
test_graph_features.py
-----------------------
Unit tests for scam_detector.features.graph_features.
"""

from __future__ import annotations

import os
import tempfile
import networkx as nx
import pytest

from scam_detector.features.graph_features import (
    build_company_infrastructure_graph,
    shared_infrastructure_flag,
    duplicate_cluster_network_size,
    export_largest_components_visualization,
)


def test_build_graph_basic() -> None:
    records = [
        {"company": "Company A", "applyLink": "https://company-a.com/apply", "internship_id": "rec1"},
        {"company": "Company B", "applyLink": "https://company-b.com/jobs", "internship_id": "rec2"},
    ]

    graph = build_company_infrastructure_graph(records)

    # Check node presence
    assert graph.has_node("company:company a")
    assert graph.has_node("company:company b")
    assert graph.has_node("domain:company-a.com")
    assert graph.has_node("domain:company-b.com")

    # Check edge presence
    assert graph.has_edge("company:company a", "domain:company-a.com")
    assert graph.has_edge("company:company b", "domain:company-b.com")


def test_build_graph_domain_exclusions() -> None:
    records = [
        {"company": "Company A", "applyLink": "https://internshala.com/apply/123", "internship_id": "rec1"},  # platform internal
        {"company": "Company B", "applyLink": "https://bit.ly/short", "internship_id": "rec2"},              # url shortener
        {"company": "Company C", "applyLink": "https://jobs.lever.co/companyc", "internship_id": "rec3"},     # known ATS
    ]

    graph = build_company_infrastructure_graph(records)

    # Company nodes must still exist
    assert graph.has_node("company:company a")
    assert graph.has_node("company:company b")
    assert graph.has_node("company:company c")

    # Excluded domain nodes must NOT exist in the graph
    assert not graph.has_node("domain:internshala.com")
    assert not graph.has_node("domain:bit.ly")
    assert not graph.has_node("domain:lever.co")


def test_build_graph_near_duplicates() -> None:
    records = [
        {"company": "Company A", "applyLink": "https://company-a.com/apply", "_id": "rec1"},
        {"company": "Company B", "applyLink": "https://company-b.com/jobs", "_id": "rec2"},
    ]

    # Mock neighbors_by_id (simulating a near-duplicate text match)
    neighbors_by_id = {
        "rec1": [("rec2", 0.95)],
        "rec2": [("rec1", 0.95)],
    }

    graph = build_company_infrastructure_graph(records, neighbors_by_id)

    # Edge between company A and company B must exist
    assert graph.has_edge("company:company a", "company:company b")


def test_shared_infrastructure_flag() -> None:
    # 4 distinct companies sharing the same domain
    records = [
        {"company": "Company 1", "applyLink": "https://sharedinfra.xyz/apply", "internship_id": "rec1"},
        {"company": "Company 2", "applyLink": "https://sharedinfra.xyz/apply", "internship_id": "rec2"},
        {"company": "Company 3", "applyLink": "https://sharedinfra.xyz/apply", "internship_id": "rec3"},
        {"company": "Company 4", "applyLink": "https://sharedinfra.xyz/apply", "internship_id": "rec4"},
        {"company": "Company 5", "applyLink": "https://cleancompany.com/apply", "internship_id": "rec5"},
    ]

    graph = build_company_infrastructure_graph(records)

    # All companies sharing the infrastructure should trigger the flag
    # because the domain is shared with 3+ OTHER distinctly-named companies (Company 1 has Company 2, 3, 4 connected)
    assert shared_infrastructure_flag("Company 1", graph) is True
    assert shared_infrastructure_flag("Company 2", graph) is True
    assert shared_infrastructure_flag("Company 3", graph) is True
    assert shared_infrastructure_flag("Company 4", graph) is True

    # Clean company should not trigger the flag
    assert shared_infrastructure_flag("Company 5", graph) is False

    # Check threshold of 3 OTHER companies: if only 3 companies share it (2 other companies)
    records_under_threshold = [
        {"company": "Company 1", "applyLink": "https://sharedinfra.xyz/apply", "internship_id": "rec1"},
        {"company": "Company 2", "applyLink": "https://sharedinfra.xyz/apply", "internship_id": "rec2"},
        {"company": "Company 3", "applyLink": "https://sharedinfra.xyz/apply", "internship_id": "rec3"},
    ]
    graph_under_threshold = build_company_infrastructure_graph(records_under_threshold)
    # Each company shares with 2 other companies, which is less than 3 other companies
    assert shared_infrastructure_flag("Company 1", graph_under_threshold) is False


def test_duplicate_cluster_network_size() -> None:
    records = [
        {"company": "Company A", "applyLink": "https://sharedinfra.xyz/apply", "internship_id": "rec1"},
        {"company": "Company B", "applyLink": "https://sharedinfra.xyz/apply", "internship_id": "rec2"},
        {"company": "Company C", "applyLink": "https://company-c.com/apply", "internship_id": "rec3"},
    ]

    neighbors_by_id = {
        "rec2": [("rec3", 0.93)],
        "rec3": [("rec2", 0.93)],
    }

    # Connections:
    # A -> domain -> B -> (text match) -> C
    # This forms a single connected component containing A, B, C.
    graph = build_company_infrastructure_graph(records, neighbors_by_id)

    assert duplicate_cluster_network_size("Company A", graph) == 3
    assert duplicate_cluster_network_size("Company B", graph) == 3
    assert duplicate_cluster_network_size("Company C", graph) == 3

    # Unknown company
    assert duplicate_cluster_network_size("Company X", graph) == 1


def test_export_largest_components_visualization() -> None:
    records = [
        {"company": "Company A", "applyLink": "https://sharedinfra.xyz/apply", "internship_id": "rec1"},
        {"company": "Company B", "applyLink": "https://sharedinfra.xyz/apply", "internship_id": "rec2"},
    ]
    graph = build_company_infrastructure_graph(records)

    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "graph.png")
        graphml_path = os.path.join(tmpdir, "graph.graphml")

        # Runs without raising exceptions
        export_largest_components_visualization(
            graph, output_image_path=img_path, output_graphml_path=graphml_path, top_n=2
        )

        assert os.path.exists(img_path)
        assert os.path.exists(graphml_path)


def test_compute_graph_network_metrics() -> None:
    from scam_detector.features.graph_features import compute_graph_network_metrics

    # Empty graph
    empty_m = compute_graph_network_metrics(nx.Graph())
    assert empty_m["total_nodes"] == 0
    assert empty_m["company_count"] == 0

    records = [
        {"company": "Company 1", "applyLink": "https://sharedinfra.xyz/apply", "internship_id": "rec1"},
        {"company": "Company 2", "applyLink": "https://sharedinfra.xyz/apply", "internship_id": "rec2"},
        {"company": "Company 3", "applyLink": "https://sharedinfra.xyz/apply", "internship_id": "rec3"},
        {"company": "Company 4", "applyLink": "https://sharedinfra.xyz/apply", "internship_id": "rec4"},
    ]
    graph = build_company_infrastructure_graph(records)
    metrics = compute_graph_network_metrics(graph)

    assert metrics["company_count"] == 4
    assert metrics["domain_count"] == 1
    assert metrics["total_nodes"] == 5
    assert metrics["multi_company_networks_count"] == 1


def test_company_network_risk_profile() -> None:
    from scam_detector.features.graph_features import company_network_risk_profile

    records = [
        {"company": "Company 1", "applyLink": "https://sharedinfra.xyz/apply", "internship_id": "rec1"},
        {"company": "Company 2", "applyLink": "https://sharedinfra.xyz/apply", "internship_id": "rec2"},
        {"company": "Company 3", "applyLink": "https://sharedinfra.xyz/apply", "internship_id": "rec3"},
        {"company": "Company 4", "applyLink": "https://sharedinfra.xyz/apply", "internship_id": "rec4"},
    ]
    graph = build_company_infrastructure_graph(records)

    prof = company_network_risk_profile("Company 1", graph)
    assert prof["is_in_graph"] is True
    assert prof["shared_infrastructure"] is True
    assert prof["network_cluster_size"] == 4
    assert "sharedinfra.xyz" in prof["connected_domains"]

    unknown = company_network_risk_profile("Unknown Org", graph)
    assert unknown["is_in_graph"] is False
    assert unknown["network_cluster_size"] == 1
