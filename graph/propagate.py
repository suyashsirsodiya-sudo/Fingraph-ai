import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from collections import deque
import networkx as nx

import config

TIER_RANK = {
    "fact": 3,
    "inference": 2,
    "hypothesis": 1
}
RANK_TO_TIER = {
    3: "fact",
    2: "inference",
    1: "hypothesis"
}

SECTOR_MAP = {
    "DAL": "Airlines",
    "FDX": "Freight & Logistics",
    "UNP": "Railroads & Transportation",
    "VLO": "Petroleum Refining",
    "XOM": "Integrated Oil & Gas",
    "F": "Automotive Manufacturing"
}


def load_shock_config(config_path: Optional[Path] = None) -> Dict[str, List[str]]:
    """Load shock_type relation filter mapping."""
    if config_path is None:
        config_path = config.BASE_DIR / "graph" / "shock_config.json"
    if config_path.exists():
        try:
            return json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"unknown": ["*"]}


def load_graph_from_edges(edges: List[Dict[str, Any]]) -> nx.DiGraph:
    """Build NetworkX DiGraph from list of edge dicts."""
    g = nx.DiGraph()
    for edge in edges:
        src = edge.get("src", "").strip()
        dst = edge.get("dst", "").strip()
        if not src or not dst:
            continue

        company = edge.get("company", dst if dst in SECTOR_MAP else "")
        sector = SECTOR_MAP.get(company, "Energy & Commodities" if "oil" in src.lower() or "fuel" in src.lower() else "General")

        # Add node attributes
        if src not in g:
            g.add_node(src, label=src, sector=SECTOR_MAP.get(src, "Energy & Commodities"))
        if dst not in g:
            g.add_node(dst, label=dst, company=company, sector=sector)

        # Add edge with all primary evidence and metadata
        g.add_edge(
            src,
            dst,
            id=edge.get("id", f"{src}->{dst}"),
            relation=edge.get("relation", "is exposed to"),
            tier=edge.get("tier", "inference"),
            confidence=float(edge.get("confidence", 0.7)),
            evidence_quote=edge.get("evidence_quote", ""),
            source_url=edge.get("source_url", ""),
            source_date=edge.get("source_date", "unknown"),
            company=company
        )
    return g


def find_matching_start_nodes(graph: nx.DiGraph, event_query: str) -> List[str]:
    """Resolve event query string to graph source nodes."""
    query = event_query.lower().replace("_", " ").strip()
    nodes = list(graph.nodes())

    # 1. Exact match
    for n in nodes:
        if n.lower() == query:
            return [n]

    # 2. Substring match
    matches = [n for n in nodes if query in n.lower() or n.lower() in query]
    if matches:
        return matches

    # 3. Keyword matching for common events like crude oil or oil price spike
    keywords = [w for w in query.split() if w not in ("price", "spike", "shock", "event")]
    if keywords:
        kw_matches = [n for n in nodes if any(kw in n.lower() for kw in keywords)]
        if kw_matches:
            return kw_matches

    return []


def propagate_impact(
    graph: nx.DiGraph,
    event: str,
    shock_type: str = "unknown",
    magnitude: float = 1.0,
    decay: float = 0.6,
    max_hops: int = 3,
    shock_config_path: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    Weighted BFS propagation:
    - Decay: 0.6 per hop (or configured)
    - Max hops: 3
    - shock_type: 'demand' | 'supply' | 'unknown' (filters edges according to editable shock_config.json)
    - Returns per node score, hop, weakest tier on path, and full path.
    """
    if graph.number_of_nodes() == 0:
        return []

    shock_config = load_shock_config(shock_config_path)
    allowed_relations = shock_config.get(shock_type.lower(), shock_config.get("unknown", ["*"]))
    allow_all_relations = "*" in allowed_relations

    start_nodes = find_matching_start_nodes(graph, event)
    if not start_nodes:
        return []

    # Best path state per node: node -> (score, hop, weakest_tier_name, full_path)
    best_results: Dict[str, Dict[str, Any]] = {}

    # Queue items: (current_node, current_score, current_hop, path_list, min_tier_rank)
    queue = deque()
    for s_node in start_nodes:
        # Base start node
        queue.append((s_node, float(magnitude), 0, [s_node], 3))

    while queue:
        curr_node, curr_score, curr_hop, path, min_rank = queue.popleft()

        # Record node if not root start node
        if curr_hop > 0:
            tier_name = RANK_TO_TIER.get(min_rank, "inference")
            if curr_node not in best_results or curr_score > best_results[curr_node]["score"]:
                company = graph.nodes[curr_node].get("company", curr_node)
                sector = SECTOR_MAP.get(company, graph.nodes[curr_node].get("sector", "Sector"))
                best_results[curr_node] = {
                    "node": curr_node,
                    "score": round(curr_score, 4),
                    "hop": curr_hop,
                    "weakest_tier": tier_name,
                    "full_path": path,
                    "company": company,
                    "sector": sector
                }

        if curr_hop >= max_hops:
            continue

        for _, next_node, edge_data in graph.out_edges(curr_node, data=True):
            relation = edge_data.get("relation", "")
            if not allow_all_relations and relation not in allowed_relations:
                continue

            conf = float(edge_data.get("confidence", 0.7))
            next_hop = curr_hop + 1
            next_score = curr_score * decay * conf

            tier = edge_data.get("tier", "inference")
            edge_rank = TIER_RANK.get(tier, 2)
            next_min_rank = min(min_rank, edge_rank)

            # Prevent infinite loops along identical path
            if next_node in path:
                continue

            # Only explore if promising
            existing = best_results.get(next_node)
            if existing is None or next_score > existing["score"]:
                queue.append((next_node, next_score, next_hop, path + [next_node], next_min_rank))

    # Return results sorted by score descending
    results = sorted(best_results.values(), key=lambda x: x["score"], reverse=True)
    return results
