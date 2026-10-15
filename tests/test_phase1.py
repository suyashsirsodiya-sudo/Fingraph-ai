"""
FinGraph AI - Phase 1 Test Suite
Tests pure graph engine, entity resolution, and database constraints.
Offline only, zero external network calls.
"""

from datetime import datetime, timezone
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from database import Base
import graph_engine
from models import Article, Entity, Evidence, Relationship
import resolution
import seed_data


# In-memory SQLite session with foreign keys enabled for constraint testing
@pytest.fixture
def in_memory_db():
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Pre-seed minimal article and evidence for tests
    article = Article(
        id="art_test",
        title="Test Article",
        source_name="Test Wire",
        retrieval_status="Source Retrieved",
        review_status="Evidence Reviewed",
        is_demo=True,
    )
    ev = Evidence(
        id="ev_test",
        article_id="art_test",
        quote="Test quote evidence.",
        quote_start=0,
        quote_end=19,
    )
    ent_a = Entity(id="ent_A", canonical_name="Node A", entity_type="company", is_demo=True)
    ent_b = Entity(id="ent_B", canonical_name="Node B", entity_type="company", is_demo=True)

    session.add_all([article, ev, ent_a, ent_b])
    session.commit()

    try:
        yield session
    finally:
        session.close()


# 1. GOLDEN FIXTURE TEST
def test_golden_fixture_positive_and_negative_propagation():
    """
    Golden fixture from specification:
    A->B (weight 1.0, confidence 0.8, pressure, direct)
    B->C (weight 0.8, confidence 0.5, pressure, direct)
    propagate(A, +10) must give B = 8.0 and C = 1.92.
    With -10, effects flip to benefit.
    """
    entities = [
        {"id": "A", "canonical_name": "Entity A", "entity_type": "company"},
        {"id": "B", "canonical_name": "Entity B", "entity_type": "company"},
        {"id": "C", "canonical_name": "Entity C", "entity_type": "company"},
    ]
    relationships = [
        {
            "id": "rel_A_B",
            "source_entity_id": "A",
            "target_entity_id": "B",
            "weight": 1.0,
            "confidence": 0.8,
            "direction": "pressure",
            "direct_or_indirect": "direct",
            "observed_or_inferred": "observed",
            "mechanism": "cost",
        },
        {
            "id": "rel_B_C",
            "source_entity_id": "B",
            "target_entity_id": "C",
            "weight": 0.8,
            "confidence": 0.5,
            "direction": "pressure",
            "direct_or_indirect": "direct",
            "observed_or_inferred": "observed",
            "mechanism": "cost",
        },
    ]

    G = graph_engine.build_graph(entities, relationships)

    # Positive shock: +10
    result_pos = graph_engine.propagate(G, start="A", pct_change=10.0, decay=0.6)
    assert "B" in result_pos["entities"]
    assert "C" in result_pos["entities"]

    res_B_pos = result_pos["entities"]["B"]
    res_C_pos = result_pos["entities"]["C"]

    assert res_B_pos["score"] == 8.0
    assert res_B_pos["effect"] == "pressure"
    assert res_B_pos["hops"] == 1
    assert res_B_pos["directness"] == "direct"
    assert res_B_pos["basis"] == "observed"

    assert res_C_pos["score"] == 1.92
    assert res_C_pos["effect"] == "pressure"
    assert res_C_pos["hops"] == 2
    assert res_C_pos["directness"] == "direct"
    assert res_C_pos["basis"] == "observed"

    # Negative shock: -10 (flips pressure -> benefit)
    result_neg = graph_engine.propagate(G, start="A", pct_change=-10.0, decay=0.6)
    res_B_neg = result_neg["entities"]["B"]
    res_C_neg = result_neg["entities"]["C"]

    assert res_B_neg["score"] == 8.0
    assert res_B_neg["effect"] == "benefit"
    assert res_C_neg["score"] == 1.92
    assert res_C_neg["effect"] == "benefit"


# 2. CYCLE IMMUNITY TEST
def test_cycle_immunity_never_repeats_nodes():
    """
    A->B plus B->A (cycle). From A, path exploration returns only [A, B]
    and never loops infinitely or repeats nodes.
    """
    entities = [
        {"id": "A", "canonical_name": "Node A"},
        {"id": "B", "canonical_name": "Node B"},
    ]
    relationships = [
        {
            "id": "rel_AB",
            "source_entity_id": "A",
            "target_entity_id": "B",
            "weight": 1.0,
            "confidence": 0.9,
            "direction": "pressure",
            "direct_or_indirect": "direct",
        },
        {
            "id": "rel_BA",
            "source_entity_id": "B",
            "target_entity_id": "A",
            "weight": 1.0,
            "confidence": 0.9,
            "direction": "pressure",
            "direct_or_indirect": "direct",
        },
    ]

    G = graph_engine.build_graph(entities, relationships)
    paths = graph_engine.find_paths(G, start="A", max_depth=5)

    assert len(paths) == 1
    p = paths[0]
    assert p["target"] == "B"
    assert p["node_sequence"] == ["A", "B"]
    # Verify no path contains duplicate nodes
    for path in paths:
        assert len(path["node_sequence"]) == len(set(path["node_sequence"]))


# 3. DEPTH COVERAGE MONOTONICITY TEST
def test_raising_depth_never_lowers_coverage():
    """Raising search depth never lowers reachable entity coverage."""
    demo_entities = seed_data.get_demo_entities()
    demo_relationships = seed_data.get_demo_relationships()
    G = graph_engine.build_graph(demo_entities, demo_relationships)

    start_node = "ent_event_hormuz"
    prop_d1 = graph_engine.propagate(G, start=start_node, pct_change=10.0, max_depth=1)
    prop_d2 = graph_engine.propagate(G, start=start_node, pct_change=10.0, max_depth=2)
    prop_d3 = graph_engine.propagate(G, start=start_node, pct_change=10.0, max_depth=3)

    count_1 = len(prop_d1["entities"])
    count_2 = len(prop_d2["entities"])
    count_3 = len(prop_d3["entities"])

    assert count_1 <= count_2 <= count_3
    assert count_3 > 0


# 4. POINT-IN-TIME (AS_OF) FILTERING TEST
def test_point_in_time_excludes_out_of_bounds_edges():
    """as_of before valid_from or after valid_to excludes the edge."""
    entities = [
        {"id": "A", "canonical_name": "Node A"},
        {"id": "B", "canonical_name": "Node B"},
    ]
    relationships = [
        {
            "id": "rel_pit",
            "source_entity_id": "A",
            "target_entity_id": "B",
            "weight": 1.0,
            "confidence": 0.9,
            "direction": "pressure",
            "valid_from": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "valid_to": datetime(2026, 12, 31, tzinfo=timezone.utc),
        }
    ]

    # Before valid_from -> excluded
    G_before = graph_engine.build_graph(entities, relationships, as_of="2025-06-01T00:00:00Z")
    assert G_before.number_of_edges() == 0

    # Within valid window -> included
    G_during = graph_engine.build_graph(entities, relationships, as_of="2026-06-01T00:00:00Z")
    assert G_during.number_of_edges() == 1

    # After valid_to -> excluded
    G_after = graph_engine.build_graph(entities, relationships, as_of="2027-01-01T00:00:00Z")
    assert G_after.number_of_edges() == 0


# 5. CONFLICT DETECTION TEST
def test_detect_conflicts_flags_opposing_edges_and_applies_penalty():
    """Opposite-direction parallel edges are reported as conflict and penalized."""
    entities = [
        {"id": "NG", "canonical_name": "Natural Gas"},
        {"id": "UT", "canonical_name": "Utilities"},
    ]
    relationships = [
        {
            "id": "rel_c1",
            "source_entity_id": "NG",
            "target_entity_id": "UT",
            "direction": "pressure",
            "confidence": 0.80,
            "evidence_id": "ev_1",
        },
        {
            "id": "rel_c2",
            "source_entity_id": "NG",
            "target_entity_id": "UT",
            "direction": "benefit",
            "confidence": 0.80,
            "evidence_id": "ev_2",
        },
    ]

    G = graph_engine.build_graph(entities, relationships)
    conflicts = graph_engine.detect_conflicts(G, penalty_factor=0.7)

    assert len(conflicts) == 1
    c = conflicts[0]
    assert c["source"] == "NG"
    assert c["target"] == "UT"
    assert c["penalty_factor"] == 0.7

    edge_1 = G.get_edge_data("NG", "UT")["rel_c1"]
    edge_2 = G.get_edge_data("NG", "UT")["rel_c2"]
    assert edge_1["conflicted"] is True
    assert edge_2["conflicted"] is True
    assert edge_1["effective_confidence"] == round(0.80 * 0.7, 4)
    assert edge_2["effective_confidence"] == round(0.80 * 0.7, 4)


# 6. ENTITY RESOLUTION TESTS
def test_entity_resolution_rules():
    """
    Tests:
    - Whole-word, case-insensitive for names
    - Exact case for tickers
    - Ambiguous mentions return candidates and resolve nothing
    """
    resolver = resolution.EntityResolver(seed_data.get_demo_entities())

    # Whole-word, case-insensitive match on name
    res_crude = resolver.resolve_mention("crude oil")
    assert res_crude["status"] == "resolved"
    assert res_crude["entity"]["id"] == "ent_crude_oil"

    # Exact case on ticker
    res_dal_exact = resolver.resolve_mention("DAL")
    assert res_dal_exact["status"] == "resolved"
    assert res_dal_exact["entity"]["id"] == "ent_co_dal"

    # Lowercase ticker should NOT resolve via ticker
    # ("dal" is not a canonical name or alias, so unresolved)
    res_dal_lower = resolver.resolve_mention("dal")
    assert res_dal_lower["status"] == "unresolved"

    # Ambiguous alias test
    ambiguous_entities = [
        {"id": "ent_1", "canonical_name": "First Energy", "aliases": ["Energy"]},
        {"id": "ent_2", "canonical_name": "Second Energy", "aliases": ["Energy"]},
    ]
    amb_resolver = resolution.EntityResolver(ambiguous_entities)
    amb_res = amb_resolver.resolve_mention("Energy")
    assert amb_res["status"] == "ambiguous"
    assert amb_res["entity"] is None
    assert len(amb_res["candidates"]) == 2


# 7. SCHEMA CONSTRAINTS TESTS
def test_observed_edge_without_evidence_is_rejected(in_memory_db):
    """An observed edge without evidence_id MUST be rejected by database check constraint."""
    bad_rel = Relationship(
        id="rel_bad_observed",
        source_entity_id="ent_A",
        target_entity_id="ent_B",
        relationship_type="supply",
        mechanism="supply",
        direction="pressure",
        explanation="Hypothesized observed relationship lacking evidence.",
        confidence=0.8,
        weight=0.8,
        direct_or_indirect="direct",
        observed_or_inferred="observed",  # Observed!
        evidence_id=None,  # Missing evidence! Must violate constraint
        is_demo=True,
    )
    in_memory_db.add(bad_rel)
    with pytest.raises(IntegrityError):
        in_memory_db.commit()
    in_memory_db.rollback()


def test_self_loop_is_rejected(in_memory_db):
    """No self-loops constraint check."""
    self_loop_rel = Relationship(
        id="rel_self_loop",
        source_entity_id="ent_A",
        target_entity_id="ent_A",  # Self loop
        relationship_type="ownership",
        mechanism="ownership",
        direction="pressure",
        explanation="Self loop relationship",
        confidence=0.8,
        weight=0.8,
        direct_or_indirect="direct",
        observed_or_inferred="inferred",
        is_demo=True,
    )
    in_memory_db.add(self_loop_rel)
    with pytest.raises(IntegrityError):
        in_memory_db.commit()
    in_memory_db.rollback()


def test_confidence_out_of_range_is_rejected(in_memory_db):
    """Confidence out of range [0.0, 1.0] is rejected."""
    out_of_range_rel = Relationship(
        id="rel_bad_conf",
        source_entity_id="ent_A",
        target_entity_id="ent_B",
        relationship_type="cost",
        mechanism="cost",
        direction="pressure",
        explanation="Invalid confidence",
        confidence=1.5,  # > 1.0
        weight=0.5,
        direct_or_indirect="direct",
        observed_or_inferred="inferred",
        is_demo=True,
    )
    in_memory_db.add(out_of_range_rel)
    with pytest.raises(IntegrityError):
        in_memory_db.commit()
    in_memory_db.rollback()


# 8. GRAPH QUALITY REPORT AUDIT TEST
def test_quality_report_on_seeded_graph():
    """Verify graph quality metrics on the demo dataset."""
    entities = seed_data.get_demo_entities()
    relationships = seed_data.get_demo_relationships()
    G = graph_engine.build_graph(entities, relationships)

    report = graph_engine.quality_report(G)
    assert report["node_count"] == 25
    assert report["edge_count"] == 41
    assert report["edges_with_evidence_pct"] > 60.0
    assert report["conflict_count"] >= 1
    assert report["average_confidence"] > 0.70
    assert report["observed_edges_count"] > 0
    assert report["inferred_edges_count"] > 0
