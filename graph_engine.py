"""
FinGraph AI - Graph Engine (Pure Module)
Zero web or DB framework dependencies. Fully unit-testable.
Implements:
1. build_graph (Point-in-time filtering)
2. find_paths (Simple paths, deterministic, capped)
3. propagate (Mathematical relative exposure scores, directional flipping, basis aggregation)
4. detect_conflicts (Opposing directional evidence penalty)
5. quality_report (Structural and evidence integrity metrics)
"""

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import networkx as nx

# Documented confidence penalty applied when opposing directional evidence is detected
CONFLICT_CONFIDENCE_PENALTY_FACTOR = 0.70


def _parse_datetime(dt_val: Any) -> Optional[datetime]:
    """Parse various datetime inputs to timezone-aware UTC datetime."""
    if dt_val is None:
        return None
    if isinstance(dt_val, datetime):
        if dt_val.tzinfo is None:
            return dt_val.replace(tzinfo=timezone.utc)
        return dt_val.astimezone(timezone.utc)
    if isinstance(dt_val, str):
        cleaned = dt_val.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(cleaned)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            return None
    return None


def build_graph(
    entities: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    as_of: Optional[Any] = None,
) -> nx.MultiDiGraph:
    """
    Construct a MultiDiGraph of entities and relationships.
    Point-in-time rule:
    If as_of is provided, include only edges where valid_from <= as_of < valid_to.
    Missing/null valid_from is open on left (-inf).
    Missing/null valid_to is open on right (+inf).
    Information dated strictly after as_of is never included.
    """
    G = nx.MultiDiGraph()
    as_of_dt = _parse_datetime(as_of)

    # Add entities as nodes
    for entity in entities:
        ent_id = str(entity.get("id"))
        ent_data = dict(entity)
        G.add_node(ent_id, **ent_data)

    # Add relationships as directed multi-edges
    for rel in relationships:
        src = str(rel.get("source_entity_id"))
        tgt = str(rel.get("target_entity_id"))
        edge_id = str(rel.get("id", f"{src}->{tgt}"))

        # Point-in-time filtering
        if as_of_dt is not None:
            v_from = _parse_datetime(rel.get("valid_from"))
            v_to = _parse_datetime(rel.get("valid_to"))

            # Exclude if edge is not yet valid at as_of
            if v_from is not None and v_from > as_of_dt:
                continue
            # Exclude if edge has already expired as of as_of
            if v_to is not None and as_of_dt >= v_to:
                continue

        edge_data = dict(rel)
        # Ensure default effective confidence equals baseline confidence
        edge_data["effective_confidence"] = float(rel.get("confidence", 1.0))
        edge_data["conflicted"] = bool(rel.get("conflicted", False))

        G.add_edge(src, tgt, key=edge_id, **edge_data)

    return G


def find_paths(
    graph: nx.MultiDiGraph,
    start: str,
    max_depth: int = 3,
    max_paths: int = 100,
    min_confidence: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Enumerate simple paths originating from start.
    A node never repeats in a path, ensuring cycle immunity.
    Order is deterministic and output is capped at max_paths.
    A path is direct only if every edge is direct.
    """
    if start not in graph:
        return []

    collected_paths: List[Dict[str, Any]] = []

    def dfs(
        curr_node: str,
        visited_nodes: List[str],
        current_edges: List[Tuple[str, str, str, Dict[str, Any]]],
    ):
        if len(current_edges) >= max_depth:
            return

        # Deterministic sorting of neighbors
        neighbors = sorted(graph.successors(curr_node))
        for nbr in neighbors:
            if nbr in visited_nodes:
                # Cycle prevention: a node never repeats in one path
                continue

            # Deterministic sorting of parallel multi-edges by key
            edge_dict = graph.get_edge_data(curr_node, nbr) or {}
            for edge_key in sorted(edge_dict.keys()):
                edge_attr = edge_dict[edge_key]
                conf = float(edge_attr.get("effective_confidence", edge_attr.get("confidence", 1.0)))
                if conf < min_confidence:
                    continue

                new_edge_step = (curr_node, nbr, edge_key, edge_attr)
                new_edge_path = current_edges + [new_edge_step]
                new_visited = visited_nodes + [nbr]

                # Format path record
                is_direct = all(
                    str(e[3].get("direct_or_indirect", "indirect")).lower() == "direct"
                    for e in new_edge_path
                )
                min_conf = min(
                    float(e[3].get("effective_confidence", e[3].get("confidence", 1.0)))
                    for e in new_edge_path
                )
                basis = (
                    "observed"
                    if all(str(e[3].get("observed_or_inferred", "inferred")).lower() == "observed" for e in new_edge_path)
                    else "inferred"
                )

                path_record = {
                    "source": start,
                    "target": nbr,
                    "hops": len(new_edge_path),
                    "is_direct": is_direct,
                    "min_confidence": round(min_conf, 4),
                    "basis": basis,
                    "node_sequence": new_visited,
                    "edge_sequence": [e[2] for e in new_edge_path],
                    "edges": [dict(e[3]) for e in new_edge_path],
                }
                collected_paths.append(path_record)

                # Continue DFS if depth permits
                if len(new_edge_path) < max_depth:
                    dfs(nbr, new_visited, new_edge_path)

    dfs(start, [start], [])

    # Sort paths deterministically: target, hops, min_confidence desc, edge_ids
    collected_paths.sort(
        key=lambda p: (p["target"], p["hops"], -p["min_confidence"], p["edge_sequence"])
    )
    return collected_paths[:max_paths]


def propagate(
    graph: nx.MultiDiGraph,
    start: str,
    pct_change: float,
    decay: float = 0.6,
    max_depth: int = 3,
    min_confidence: float = 0.0,
) -> Dict[str, Any]:
    """
    Simulate relative event shock propagation across the graph.
    Score formula:
      score = round(abs(pct_change) * product over edges of (weight * confidence) * decay^(hops-1), 4)
    Effect rules:
      - Direction of the LAST edge in the path.
      - Swap 'pressure' and 'benefit' if pct_change is negative.
      - If any edge on path is 'unclear', label is 'unclear'.
      - If any edge is 'mixed', label is 'mixed'.
    Returns best path and top 3 alternative paths for every reachable entity.
    """
    if start not in graph:
        return {
            "label": "Scenario Analysis - Not a Price Forecast",
            "source_entity_id": start,
            "pct_change": pct_change,
            "entities": {},
        }

    raw_paths = find_paths(
        graph=graph,
        start=start,
        max_depth=max_depth,
        max_paths=1000,
        min_confidence=min_confidence,
    )

    # Group candidate paths by target entity
    entity_paths: Dict[str, List[Dict[str, Any]]] = {}

    abs_pct = abs(float(pct_change))

    for p in raw_paths:
        tgt = p["target"]
        hops = p["hops"]
        edges = p["edges"]

        # Product of (weight * confidence) along edges
        product_wc = 1.0
        for e in edges:
            w = float(e.get("weight", 1.0))
            c = float(e.get("effective_confidence", e.get("confidence", 1.0)))
            product_wc *= (w * c)

        decay_factor = decay ** (hops - 1)
        score = round(abs_pct * product_wc * decay_factor, 4)

        # Effect calculation
        has_unclear = any(str(e.get("direction", "")).lower() == "unclear" for e in edges)
        has_mixed = any(str(e.get("direction", "")).lower() == "mixed" for e in edges)

        if has_unclear:
            effect = "unclear"
        elif has_mixed:
            effect = "mixed"
        else:
            last_dir = str(edges[-1].get("direction", "unclear")).lower()
            if pct_change < 0:
                if last_dir == "pressure":
                    effect = "benefit"
                elif last_dir == "benefit":
                    effect = "pressure"
                else:
                    effect = last_dir
            else:
                effect = last_dir

        directness = "direct" if p["is_direct"] else "indirect"
        mechanisms = [e.get("mechanism", "other") for e in edges]

        # Entity names along the path
        entity_names = []
        for n_id in p["node_sequence"]:
            node_attrs = graph.nodes.get(n_id, {})
            entity_names.append(node_attrs.get("canonical_name", n_id))

        evaluated_path = {
            "score": score,
            "hops": hops,
            "directness": directness,
            "effect": effect,
            "min_confidence": p["min_confidence"],
            "basis": p["basis"],
            "path": {
                "entity_names": entity_names,
                "node_ids": p["node_sequence"],
                "edge_ids": p["edge_sequence"],
            },
            "mechanisms": mechanisms,
        }

        entity_paths.setdefault(tgt, []).append(evaluated_path)

    # Format result per entity: best path + top 3 alternatives
    entities_result: Dict[str, Any] = {}
    for tgt_id, path_candidates in entity_paths.items():
        # Sort by score descending, then hops ascending, then edge_ids
        path_candidates.sort(
            key=lambda item: (-item["score"], item["hops"], item["path"]["edge_ids"])
        )

        best = path_candidates[0]
        alternatives = path_candidates[1:4]  # Top 3 alternative paths

        tgt_node_data = graph.nodes.get(tgt_id, {})
        entities_result[tgt_id] = {
            "entity_id": tgt_id,
            "canonical_name": tgt_node_data.get("canonical_name", tgt_id),
            "ticker": tgt_node_data.get("ticker"),
            "entity_type": tgt_node_data.get("entity_type"),
            "score": best["score"],
            "hops": best["hops"],
            "directness": best["directness"],
            "effect": best["effect"],
            "min_confidence": best["min_confidence"],
            "basis": best["basis"],
            "path": best["path"],
            "mechanisms": best["mechanisms"],
            "top_alternative_paths": alternatives,
        }

    return {
        "label": "Scenario Analysis - Not a Price Forecast",
        "source_entity_id": start,
        "source_entity_name": graph.nodes.get(start, {}).get("canonical_name", start),
        "pct_change": pct_change,
        "decay": decay,
        "entities": entities_result,
    }


def detect_conflicts(
    graph: nx.MultiDiGraph,
    penalty_factor: float = CONFLICT_CONFIDENCE_PENALTY_FACTOR,
) -> List[Dict[str, Any]]:
    """
    Identify conflicting parallel edges between the same source and target pair
    having opposite directions ('pressure' vs 'benefit') from different evidence.
    Marks both edges as conflicted, lowers their effective confidence by penalty_factor,
    and returns the complete list of conflicts.
    """
    conflicts: List[Dict[str, Any]] = []

    # Iterate through all source-target pairs
    for u in graph.nodes:
        for v in graph.successors(u):
            edges_dict = graph.get_edge_data(u, v) or {}
            if len(edges_dict) < 2:
                continue

            edge_items = list(edges_dict.items())
            for i in range(len(edge_items)):
                key1, data1 = edge_items[i]
                dir1 = str(data1.get("direction", "")).lower()

                for j in range(i + 1, len(edge_items)):
                    key2, data2 = edge_items[j]
                    dir2 = str(data2.get("direction", "")).lower()

                    is_opposing = (
                        (dir1 == "pressure" and dir2 == "benefit")
                        or (dir1 == "benefit" and dir2 == "pressure")
                    )

                    ev1 = data1.get("evidence_id")
                    ev2 = data2.get("evidence_id")
                    is_diff_evidence = (ev1 != ev2) or (ev1 is None and ev2 is None)

                    if is_opposing and is_diff_evidence:
                        # Mark conflicted
                        data1["conflicted"] = True
                        data2["conflicted"] = True
                        data1["conflict_penalty_factor"] = penalty_factor
                        data2["conflict_penalty_factor"] = penalty_factor

                        base_conf1 = float(data1.get("confidence", 1.0))
                        base_conf2 = float(data2.get("confidence", 1.0))

                        data1["effective_confidence"] = round(base_conf1 * penalty_factor, 4)
                        data2["effective_confidence"] = round(base_conf2 * penalty_factor, 4)

                        conflicts.append({
                            "source": u,
                            "target": v,
                            "edge_1_id": key1,
                            "edge_1_direction": dir1,
                            "edge_1_confidence": base_conf1,
                            "edge_2_id": key2,
                            "edge_2_direction": dir2,
                            "edge_2_confidence": base_conf2,
                            "penalty_factor": penalty_factor,
                            "explanation": (
                                f"Parallel edges between '{u}' and '{v}' express opposing directions "
                                f"('{dir1}' vs '{dir2}') under different evidence. "
                                f"Effective confidence discounted by factor {penalty_factor}."
                            ),
                        })

    return conflicts


def quality_report(graph: nx.MultiDiGraph) -> Dict[str, Any]:
    """
    Generate graph structural and evidence quality audit report:
    - Node and edge counts
    - % edges with evidence
    - Orphan nodes (degree 0)
    - Weakly connected components
    - Conflict count
    - Average confidence
    - Inferred vs observed edge counts
    """
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()

    edges_with_evidence = 0
    total_confidence = 0.0
    observed_count = 0
    inferred_count = 0

    for u, v, k, data in graph.edges(keys=True, data=True):
        if data.get("evidence_id") is not None:
            edges_with_evidence += 1

        conf = float(data.get("effective_confidence", data.get("confidence", 1.0)))
        total_confidence += conf

        basis = str(data.get("observed_or_inferred", "inferred")).lower()
        if basis == "observed":
            observed_count += 1
        else:
            inferred_count += 1

    pct_with_evidence = (
        round((edges_with_evidence / edge_count) * 100.0, 2) if edge_count > 0 else 0.0
    )
    avg_confidence = round(total_confidence / edge_count, 4) if edge_count > 0 else 0.0

    orphan_nodes = [n for n in graph.nodes if graph.degree(n) == 0]
    wcc_count = nx.number_weakly_connected_components(graph) if node_count > 0 else 0

    # Tally conflicts
    conflicts = detect_conflicts(graph)

    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "edges_with_evidence_count": edges_with_evidence,
        "edges_with_evidence_pct": pct_with_evidence,
        "orphan_nodes": orphan_nodes,
        "orphan_node_count": len(orphan_nodes),
        "weakly_connected_components": wcc_count,
        "conflict_count": len(conflicts),
        "conflicts": conflicts,
        "average_confidence": avg_confidence,
        "observed_edges_count": observed_count,
        "inferred_edges_count": inferred_count,
    }
