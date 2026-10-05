"""
graph_features.py
-----------------
Graph-based analysis for iFind scam detector.
Builds a NetworkX graph of companies, applyLink domains, and duplicate text clusters
to identify coordinated posting networks.
"""

from __future__ import annotations

import logging
from typing import Any
import networkx as nx
import matplotlib.pyplot as plt

from scam_detector.features.duplicate_detection import DuplicateIndex, _record_id
from scam_detector.features.url_features import (
    parse_url_components,
    is_platform_internal_link,
    is_url_shortener,
    is_known_ats_domain,
)

log = logging.getLogger("scam_detector.features.graph")


def build_company_infrastructure_graph(
    records: list[dict[str, Any]],
    neighbors_by_id: dict[str, list[tuple[str, float]]] | None = None,
) -> nx.Graph:
    """
    Build a networkx Graph where:
      - Company nodes represent normalized company names.
      - Domain nodes represent normalized off-platform applyLink domains.
      - Edges connect companies to the domain(s) they post through.
      - Edges connect companies that share a near-duplicate text cluster.

    Excludes platform-internal domains, URL shorteners, and known ATS domains
    to prevent large hub nodes that merge unrelated networks.
    """
    graph = nx.Graph()
    if not records:
        return graph

    # Track mapping from record_id -> company name (normalized & original)
    id_to_company: dict[str, str] = {}
    for idx, r in enumerate(records):
        rid = _record_id(r, idx)
        company = (r.get("company") or "").strip()
        if company:
            id_to_company[rid] = company

    # 1. Add company nodes and their applyLink domain edges
    for idx, r in enumerate(records):
        company = (r.get("company") or "").strip()
        if not company:
            continue

        company_key = company.lower()
        company_node = f"company:{company_key}"
        graph.add_node(company_node, label=company, type="company")

        # Extract domain from applyLink / apply_link
        apply_link = r.get("applyLink") or r.get("apply_link") or ""
        if apply_link:
            # Check if this is a platform-internal, shortener, or known ATS domain
            is_internal = is_platform_internal_link(apply_link)
            is_short = is_url_shortener(apply_link)
            is_ats = is_known_ats_domain(apply_link)

            if not (is_internal or is_short or is_ats):
                components = parse_url_components(apply_link)
                registered_domain = components.get("registered_domain") or components.get("domain")
                if registered_domain:
                    domain_key = registered_domain.lower()
                    domain_node = f"domain:{domain_key}"
                    graph.add_node(domain_node, label=registered_domain, type="domain")
                    graph.add_edge(company_node, domain_node)

    # 2. Add edges between companies that share near-duplicate text clusters
    if neighbors_by_id is None:
        try:
            dup_index = DuplicateIndex()
            dup_index.build(records)
            neighbors_by_id = {}
            for idx, r in enumerate(records):
                rid = _record_id(r, idx)
                neighbors_by_id[rid] = dup_index.find_near_duplicates(rid, threshold=0.92)
        except Exception as exc:
            log.warning("DuplicateIndex building failed in graph builder (%s). Skipping near-duplicate edges.", exc)
            neighbors_by_id = {}

    for rid, neighbors in neighbors_by_id.items():
        co_a = id_to_company.get(rid)
        if not co_a:
            continue
        
        co_key_a = co_a.lower()
        company_node_a = f"company:{co_key_a}"

        for neighbor_id, _sim in neighbors:
            co_b = id_to_company.get(neighbor_id)
            if not co_b:
                continue
            
            co_key_b = co_b.lower()
            if co_key_a != co_key_b:
                company_node_b = f"company:{co_key_b}"
                graph.add_edge(company_node_a, company_node_b)

    return graph


def shared_infrastructure_flag(company: str, graph: nx.Graph) -> bool:
    """
    Return True if this company's applyLink domain is shared with 3+ OTHER distinctly-named
    companies (case-insensitive distinct names).
    """
    if not company:
        return False

    company_key = company.strip().lower()
    company_node = f"company:{company_key}"

    if not graph.has_node(company_node):
        return False

    for neighbor in graph.neighbors(company_node):
        if neighbor.startswith("domain:"):
            # Find other company nodes connected to this domain
            other_companies = {
                n for n in graph.neighbors(neighbor)
                if n.startswith("company:") and n != company_node
            }
            if len(other_companies) >= 3:
                return True

    return False


def duplicate_cluster_network_size(company: str, graph: nx.Graph) -> int:
    """
    Return the size of the connected component this company belongs to
    (counting only company nodes). Returns 1 if company is a singleton or not in graph.
    """
    if not company:
        return 1

    company_key = company.strip().lower()
    company_node = f"company:{company_key}"

    if not graph.has_node(company_node):
        return 1

    component = nx.node_connected_component(graph, company_node)
    company_nodes = [n for n in component if n.startswith("company:")]
    return len(company_nodes)


def export_largest_components_visualization(
    graph: nx.Graph,
    output_image_path: str,
    output_graphml_path: str | None = None,
    top_n: int = 3,
) -> None:
    """
    Export a visual representation of the largest connected components in the graph
    to a static image using matplotlib, and optionally save as GraphML.
    """
    # 1. Save GraphML if requested
    if output_graphml_path:
        nx.write_graphml(graph, output_graphml_path)

    # 2. Extract top_n largest components
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    if not components:
        plt.figure(figsize=(8, 6))
        plt.title("No components found")
        plt.savefig(output_image_path, bbox_inches="tight")
        plt.close()
        return

    top_components = components[:top_n]
    subgraph_nodes = set()
    for comp in top_components:
        subgraph_nodes.update(comp)
    
    subgraph = graph.subgraph(subgraph_nodes)
    if len(subgraph) == 0:
        plt.figure(figsize=(8, 6))
        plt.title("Empty Subgraph")
        plt.savefig(output_image_path, bbox_inches="tight")
        plt.close()
        return

    # Draw the subgraph
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(subgraph, k=0.25, seed=42)

    company_nodes = [n for n in subgraph.nodes() if n.startswith("company:")]
    domain_nodes = [n for n in subgraph.nodes() if n.startswith("domain:")]

    labels = {}
    for n in subgraph.nodes():
        node_label = subgraph.nodes[n].get("label", n)
        labels[n] = node_label

    # Draw company nodes
    nx.draw_networkx_nodes(
        subgraph,
        pos,
        nodelist=company_nodes,
        node_color="#1f77b4",
        node_size=800,
        alpha=0.85,
        label="Companies",
    )

    # Draw domain nodes
    nx.draw_networkx_nodes(
        subgraph,
        pos,
        nodelist=domain_nodes,
        node_color="#ff7f0e",
        node_size=600,
        alpha=0.85,
        node_shape="s",
        label="Domains",
    )

    # Draw edges
    nx.draw_networkx_edges(subgraph, pos, width=1.5, alpha=0.6, edge_color="#aaaaaa")

    # Draw labels
    nx.draw_networkx_labels(
        subgraph, pos, labels, font_size=8, font_weight="bold", font_family="sans-serif"
    )

    plt.title(
        f"Top {min(top_n, len(top_components))} Coordinated Posting Networks",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(scatterpoints=1, loc="best")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300, bbox_inches="tight")
    plt.close()


def compute_graph_network_metrics(graph: nx.Graph) -> dict[str, Any]:
    """
    Compute aggregate graph topological metrics across all company and domain nodes.
    Useful for system-level monitoring and cluster reporting.
    """
    total_nodes = graph.number_of_nodes()
    total_edges = graph.number_of_edges()
    if total_nodes == 0:
        return {
            "total_nodes": 0,
            "total_edges": 0,
            "company_count": 0,
            "domain_count": 0,
            "connected_components_count": 0,
            "largest_component_size": 0,
            "density": 0.0,
            "multi_company_networks_count": 0,
        }

    company_nodes = [n for n in graph.nodes if n.startswith("company:")]
    domain_nodes = [n for n in graph.nodes if n.startswith("domain:")]
    components = list(nx.connected_components(graph))
    
    multi_company_count = sum(
        1 for comp in components if len([n for n in comp if n.startswith("company:")]) >= 3
    )
    largest_comp_size = max((len(c) for c in components), default=0)
    density = float(nx.density(graph))

    return {
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "company_count": len(company_nodes),
        "domain_count": len(domain_nodes),
        "connected_components_count": len(components),
        "largest_component_size": largest_comp_size,
        "density": round(density, 6),
        "multi_company_networks_count": multi_company_count,
    }


def company_network_risk_profile(company: str, graph: nx.Graph) -> dict[str, Any]:
    """
    Compute a detailed graph risk profile for a specific company entity.
    """
    if not company:
        return {
            "company": "",
            "is_in_graph": False,
            "degree": 0,
            "shared_infrastructure": False,
            "network_cluster_size": 1,
            "connected_domains": [],
            "connected_peer_companies": [],
        }

    company_key = company.strip().lower()
    company_node = f"company:{company_key}"

    if not graph.has_node(company_node):
        return {
            "company": company,
            "is_in_graph": False,
            "degree": 0,
            "shared_infrastructure": False,
            "network_cluster_size": 1,
            "connected_domains": [],
            "connected_peer_companies": [],
        }

    degree = graph.degree(company_node)
    shared_infra = shared_infrastructure_flag(company, graph)
    network_size = duplicate_cluster_network_size(company, graph)

    connected_domains = [
        graph.nodes[n].get("label", n.replace("domain:", ""))
        for n in graph.neighbors(company_node)
        if n.startswith("domain:")
    ]
    
    # Peer companies directly or 2-hop connected
    component = nx.node_connected_component(graph, company_node)
    peer_companies = [
        graph.nodes[n].get("label", n.replace("company:", ""))
        for n in component
        if n.startswith("company:") and n != company_node
    ]

    return {
        "company": company,
        "is_in_graph": True,
        "degree": int(degree),
        "shared_infrastructure": shared_infra,
        "network_cluster_size": network_size,
        "connected_domains": connected_domains,
        "connected_peer_companies": peer_companies[:10],
    }
