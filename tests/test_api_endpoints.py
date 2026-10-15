"""
FinGraph AI - Automated Test Suite for API, Graph Engine, and Database Contracts
Runs fully offline with pytest and httpx/TestClient.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app
import graph_engine
import seed_data


@pytest.fixture
def client():
    return TestClient(app)


def test_api_status_honesty_rules(client):
    """Status must report demo mode and false for live_data_connected."""
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "demo"
    assert data["mode"] == "demo"
    assert data["live_data_connected"] is False
    assert "DEMO" in data["label"]


def test_api_overview_kpis(client):
    """Overview KPIs must be derived from database records."""
    res = client.get("/api/overview")
    assert res.status_code == 200
    data = res.json()
    assert data["events_analyzed"] >= 1
    assert data["entities_connected"] >= 20
    assert data["impact_paths"] > 0
    assert data["sources_verified_pct"] is not None
    assert data["change_pct"] is None  # Never fake numbers


def test_api_events_list(client):
    """Events endpoint lists maintained events."""
    res = client.get("/api/events")
    assert res.status_code == 200
    events = res.json()
    assert len(events) >= 1
    ev_ids = [e["id"] for e in events]
    assert "evt_demo_hormuz" in ev_ids or any("hormuz" in e_id.lower() or "crude" in e_id.lower() for e_id in ev_ids)


def test_api_graph_cytoscape_format_no_dangling_edges(client):
    """
    Graph endpoint must return valid Cytoscape elements
    and guarantee NO dangling edge endpoints.
    """
    res = client.get("/api/graph?max_depth=3")
    assert res.status_code == 200
    data = res.json()
    assert "nodes" in data
    assert "edges" in data
    assert "metadata" in data

    nodes = data["nodes"]
    edges = data["edges"]

    assert len(nodes) > 0
    assert len(edges) > 0

    node_ids = {n["data"]["id"] for n in nodes}

    # Verify no dangling edges
    for e in edges:
        src = e["data"]["source"]
        tgt = e["data"]["target"]
        assert src in node_ids, f"Dangling edge source {src} not in nodes"
        assert tgt in node_ids, f"Dangling edge target {tgt} not in nodes"
        assert e["data"]["provenance"] == "demo"
        assert "explanation" in e["data"]


def test_api_graph_filters(client):
    """Direct and indirect edge filters work."""
    res_direct = client.get("/api/graph?directness=direct")
    assert res_direct.status_code == 200
    edges_direct = res_direct.json()["edges"]
    for e in edges_direct:
        assert e["data"]["directness"] == "direct"

    res_indirect = client.get("/api/graph?directness=indirect")
    assert res_indirect.status_code == 200
    edges_indirect = res_indirect.json()["edges"]
    for e in edges_indirect:
        assert e["data"]["directness"] == "indirect"


def test_api_node_and_edge_detail(client):
    """Detailed node and edge lookups."""
    # Node detail
    res_node = client.get("/api/nodes/ent_crude_oil")
    assert res_node.status_code == 200
    node_data = res_node.json()
    assert node_data["canonical_name"] == "Crude Oil"
    assert node_data["entity_type"] == "commodity"

    # Edge detail
    res_edge = client.get("/api/edges/rel_01")
    assert res_edge.status_code == 200
    edge_data = res_edge.json()
    assert edge_data["source"] == "ent_event_hormuz"
    assert edge_data["target"] == "ent_crude_oil"
    assert "uncertainty_sentence" in edge_data


def test_api_unknown_endpoints_return_404(client):
    """404 returned for nonexistent entities."""
    res = client.get("/api/nodes/nonexistent_entity_999")
    assert res.status_code == 404
    assert "detail" in res.json()

    res = client.get("/api/edges/nonexistent_edge_999")
    assert res.status_code == 404
    assert "detail" in res.json()


def test_api_graph_conflicts_and_quality(client):
    """Graph quality and conflicts endpoints return valid metrics."""
    res_q = client.get("/api/graph/quality")
    assert res_q.status_code == 200
    q = res_q.json()
    assert q["node_count"] >= 20
    assert q["edge_count"] >= 30
    assert q["conflict_count"] >= 1

    res_c = client.get("/api/graph/conflicts")
    assert res_c.status_code == 200
    conflicts = res_c.json()
    assert len(conflicts) >= 1
    # Check conflicting pair between Natural Gas and Utilities
    sources = {c["source"] for c in conflicts}
    targets = {c["target"] for c in conflicts}
    assert "ent_nat_gas" in sources
    assert "ent_sec_utilities" in targets


def test_api_scenario_creation_and_label(client):
    """Scenario propagation returns mandatory non-predictive label."""
    payload = {
        "title": "Automated Test Crude Spike",
        "source_entity_id": "ent_event_hormuz",
        "pct_change": 12.0,
        "horizon": "Short Term",
    }
    res = client.post("/api/scenarios", json=payload)
    assert res.status_code == 200
    sc = res.json()
    assert sc["label"] == "Scenario Analysis - Not a Price Forecast"
    assert "results" in sc
    assert "entities" in sc["results"]


def test_api_scenario_get_and_delete(client):
    """Scenario detail lookup and deletion."""
    # Create scenario
    payload = {
        "title": "Temporary Test Shock",
        "source_entity_id": "ent_event_hormuz",
        "pct_change": 5.0,
        "horizon": "Short-term",
    }
    create_res = client.post("/api/scenarios", json=payload)
    assert create_res.status_code == 200
    sc_id = create_res.json()["id"]

    # Retrieve detail
    detail_res = client.get(f"/api/scenarios/{sc_id}")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert detail["title"] == "Temporary Test Shock"
    assert "results" in detail

    # Delete scenario
    del_res = client.delete(f"/api/scenarios/{sc_id}")
    assert del_res.status_code == 200
    assert del_res.json()["deleted"] == sc_id

    # Verify 404 after deletion
    detail_res2 = client.get(f"/api/scenarios/{sc_id}")
    assert detail_res2.status_code == 404


def test_api_data_quality_endpoint(client):
    """Data Quality & Lineage Center endpoint returns full audit report."""
    res = client.get("/api/data-quality")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "audited"
    assert len(data["pipeline_stages"]) == 6
    assert "metrics" in data
    m = data["metrics"]
    assert m["deduplication_rate_pct"] == 100.0
    assert m["schema_compliance_pct"] == 100.0
    assert m["missing_title_count"] == 0
    assert m["missing_source_count"] == 0
    assert "evidence_coverage_pct" in m
    assert "conflicts_detected" in m
    assert "providers" in data


def test_api_news_filters_and_sorting(client):
    """News API supports query parameters: category, source, sort, review_status."""
    # Test sort desc
    res_desc = client.get("/api/news?limit=5&sort=desc")
    assert res_desc.status_code == 200
    data_desc = res_desc.json()
    assert "articles" in data_desc
    assert "total" in data_desc

    # Test sort asc
    res_asc = client.get("/api/news?limit=5&sort=asc")
    assert res_asc.status_code == 200

    # Test source filter
    res_src = client.get("/api/news?source=all")
    assert res_src.status_code == 200

    # Test review_status filter
    res_rev = client.get("/api/news?review_status=verified")
    assert res_rev.status_code == 200


def test_api_observations_catalog(client):
    """Historical observation API returns empirical series catalog with governance disclaimer."""
    # Test Brent Crude
    res_crude = client.get("/api/observations?entity_id=ent_crude_oil")
    assert res_crude.status_code == 200
    data_crude = res_crude.json()
    assert len(data_crude["series"]) > 0
    assert data_crude["is_empirical_observation"] is True
    assert "correlation is strictly distinguished from causation" in data_crude["correlation_disclaimer"]

    # Test Delta Air Lines
    res_dal = client.get("/api/observations?entity_id=ent_dal")
    assert res_dal.status_code == 200
    data_dal = res_dal.json()
    assert len(data_dal["series"]) > 0

    # Test S&P 500
    res_spy = client.get("/api/observations?entity_id=ent_spy")
    assert res_spy.status_code == 200
    data_spy = res_spy.json()
    assert len(data_spy["series"]) > 0


def test_api_settings_persistence(client):
    """Settings PUT and GET persist new settings keys correctly."""
    # Read current
    get_res = client.get("/api/settings")
    assert get_res.status_code == 200

    # Update density and layout
    put_res = client.put("/api/settings", json={
        "density": "compact",
        "graph_layout": "cose",
        "refresh_interval": "30",
        "notifications_enabled": "true",
    })
    assert put_res.status_code == 200
    assert put_res.json()["updated"]["density"] == "compact"

    # Verify persisted in subsequent GET
    get_res2 = client.get("/api/settings")
    assert get_res2.status_code == 200
    assert get_res2.json()["density"] == "compact"
    assert get_res2.json()["graph_layout"] == "cose"

