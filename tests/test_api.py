import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
import networkx as nx

from api.main import app, load_data_into_memory
import api.main as api_module
from graph.propagate import propagate_impact

client = TestClient(app)


@pytest.fixture(autouse=True)
def ensure_data_loaded():
    """Ensure data is loaded before each test."""
    load_data_into_memory()


def test_docs_and_openapi_spec():
    """Verify /docs loads and openapi lists every required endpoint."""
    res = client.get("/docs")
    assert res.status_code == 200

    schema_res = client.get("/openapi.json")
    assert schema_res.status_code == 200
    paths = schema_res.json()["paths"]

    required_endpoints = [
        "/impact/{event}",
        "/queue",
        "/kpis",
        "/kpis/trend",
        "/edges",
        "/evidence/{edge_id}",
        "/sources",
        "/events",
        "/chat",
        "/flow",
        "/flow-view",
        "/graph-view"
    ]
    for endpoint in required_endpoints:
        assert endpoint in paths, f"Missing endpoint {endpoint} in OpenAPI schema"


def test_serve_dashboard_and_views():
    """Verify GET /, /flow-view, and /graph-view serve their respective HTML pages."""
    res_dash = client.get("/")
    assert res_dash.status_code == 200
    assert "FinGraph" in res_dash.text or "Impact suite" in res_dash.text


    res_flow = client.get("/flow-view")
    assert res_flow.status_code == 200
    assert "Evidence Flow" in res_flow.text

    res_graph = client.get("/graph-view")
    assert res_graph.status_code == 200
    assert "cytoscape" in res_graph.text.lower()


def test_flow_endpoint_columns_and_links():
    """
    Verify GET /flow builds 6 columns, every column sums to the same total,
    and each link matches its source node.
    """
    res = client.get("/flow")
    assert res.status_code == 200
    data = res.json()

    assert data["total"] > 0
    total = data["total"]

    # Verify 6 columns
    assert len(data["columns"]) == 6
    expected_col_names = ["Ingestion", "Relation type", "Evidence check", "Evidence tier", "Impact", "Validation"]
    actual_col_names = [col["name"] for col in data["columns"]]
    assert actual_col_names == expected_col_names

    # Rule: every column sums to the exact same total
    col_sums = [sum(n["value"] for n in col["nodes"]) for col in data["columns"]]
    for idx, col_sum in enumerate(col_sums):
        assert col_sum == total, f"Column {expected_col_names[idx]} sum {col_sum} does not match total {total}"

    # Rule: each link matches its source node and valid target
    all_node_ids = {n["id"] for n in data["N"]}
    node_col_map = {n["id"]: n["column"] for n in data["N"]}

    assert len(data["links"]) > 0
    for link in data["links"]:
        src = link["source"]
        tgt = link["target"]
        assert src in all_node_ids, f"Link source '{src}' missing from node list"
        assert tgt in all_node_ids, f"Link target '{tgt}' missing from node list"
        assert node_col_map[src] + 1 == node_col_map[tgt], f"Link from {src} to {tgt} does not span adjacent columns"
        assert link["value"] > 0

    # Verify link conservation per column transition
    for col_idx in range(5):
        col_links = [l for l in data["links"] if l["source_col"] == col_idx]
        link_sum = sum(l["value"] for l in col_links)
        assert link_sum == total, f"Links between col {col_idx} and {col_idx+1} sum to {link_sum}, expected {total}"


def test_impact_propagation():
    """Test /impact/{event} with shock_type, magnitude, and decay."""
    res = client.get("/impact/crude_oil?shock_type=supply&magnitude=1.0&decay=0.6&max_hops=3")
    assert res.status_code == 200
    data = res.json()
    assert data["event"] == "crude_oil"
    assert data["shock_type"] == "supply"
    assert data["magnitude"] == 1.0
    assert "nodes" in data

    if data["nodes"]:
        first = data["nodes"][0]
        assert "node" in first
        assert "score" in first
        assert "hop" in first
        assert "weakest_tier" in first
        assert "full_path" in first
        assert 1 <= first["hop"] <= 3
        assert first["weakest_tier"] in ("fact", "inference", "hypothesis")


def test_impact_demand_and_unknown_shock():
    """Test impact with different shock types."""
    res_demand = client.get("/impact/crude_oil?shock_type=demand")
    assert res_demand.status_code == 200

    res_unknown = client.get("/impact/crude_oil?shock_type=unknown")
    assert res_unknown.status_code == 200
    assert "nodes" in res_unknown.json()


def test_impact_nonexistent_event():
    """Test that querying an unknown event returns clean empty results."""
    res = client.get("/impact/completely_nonexistent_event_9999")
    assert res.status_code == 200
    data = res.json()
    assert data["nodes"] == []


def test_queue_endpoint():
    """Test /queue returns items needing human review and counts per task."""
    res = client.get("/queue")
    assert res.status_code == 200
    data = res.json()
    assert "total_needing_review" in data
    assert "counts_by_task" in data
    assert "low_confidence" in data["counts_by_task"]
    assert "hypothesis_tier" in data["counts_by_task"]
    assert "unreviewed" in data["counts_by_task"]
    assert "items" in data
    assert isinstance(data["items"], list)


def test_kpis_endpoint():
    """Test /kpis returns verified metrics and confidence stats."""
    res = client.get("/kpis")
    assert res.status_code == 200
    data = res.json()
    assert data["edge_count"] > 0
    assert data["verified_count"] > 0
    assert data["pct_verified"] > 0
    assert "confidence" in data
    assert "average" in data["confidence"]
    assert "peak" in data["confidence"]
    assert "low" in data["confidence"]
    assert 0.0 <= data["confidence"]["average"] <= 1.0


def test_kpis_trend_endpoint():
    """Test /kpis/trend?days=7 returns daily confidence numbers."""
    res = client.get("/kpis/trend?days=7")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    if data:
        item = data[0]
        assert "date" in item
        assert "average_confidence" in item
        assert "count" in item


def test_edges_filter_and_evidence():
    """Test /edges filtering and /evidence/{edge_id} lookup."""
    # 1. Fetch all edges
    res_all = client.get("/edges")
    assert res_all.status_code == 200
    edges = res_all.json()
    assert len(edges) > 0

    first_edge = edges[0]
    edge_id = first_edge["id"]

    # 2. Filter by tier
    res_tier = client.get(f"/edges?tier={first_edge['tier']}")
    assert res_tier.status_code == 200
    for e in res_tier.json():
        assert e["tier"].lower() == first_edge["tier"].lower()

    # 3. Filter by sector
    res_sector = client.get("/edges?sector=Airlines")
    assert res_sector.status_code == 200

    # 4. Fetch specific evidence
    res_ev = client.get(f"/evidence/{edge_id}")
    assert res_ev.status_code == 200
    ev_data = res_ev.json()
    assert ev_data["id"] == edge_id
    assert "evidence_quote" in ev_data
    assert "source_url" in ev_data
    assert len(ev_data["evidence_quote"]) > 0

    # 5. Nonexistent evidence gives 404
    res_404 = client.get("/evidence/non_existent_edge_id_12345")
    assert res_404.status_code == 404
    assert res_404.json()["detail"] == "Evidence for edge not found"


def test_sources_endpoint():
    """Test /sources returns last_fetched and fresh/stale flags."""
    res = client.get("/sources")
    assert res.status_code == 200
    sources = res.json()
    assert len(sources) > 0
    for s in sources:
        assert "source_name" in s
        assert "last_fetched" in s
        assert "status" in s
        assert s["status"] in ("fresh", "stale")
        assert "is_stale" in s


def test_events_endpoint():
    """Test /events returns event records with dates."""
    res = client.get("/events")
    assert res.status_code == 200
    events = res.json()
    assert len(events) > 0
    first = events[0]
    assert "event_id" in first
    assert "event_name" in first
    assert "event_date" in first


def test_empty_data_cases(monkeypatch):
    """Test all endpoints gracefully return clean empty results on empty data."""
    # 1. Empty graph propagation
    empty_graph = nx.DiGraph()
    empty_impact = propagate_impact(empty_graph, "crude_oil")
    assert empty_impact == []

    # 2. Empty edges in API
    monkeypatch.setattr(api_module, "app_edges", [])
    monkeypatch.setattr(api_module, "app_graph", nx.DiGraph())

    # /impact on empty data
    res_impact = client.get("/impact/crude_oil")
    assert res_impact.status_code == 200
    assert res_impact.json()["nodes"] == []

    # /kpis on empty data
    res_kpis = client.get("/kpis")
    assert res_kpis.status_code == 200
    kpis_data = res_kpis.json()
    assert kpis_data["edge_count"] == 0
    assert kpis_data["pct_verified"] == 0.0
    assert kpis_data["confidence"]["average"] == 0.0

    # /queue on empty data
    res_queue = client.get("/queue")
    assert res_queue.status_code == 200
    assert res_queue.json()["total_needing_review"] == 0
    assert res_queue.json()["items"] == []

    # /edges on empty data
    res_edges = client.get("/edges")
    assert res_edges.status_code == 200
    assert res_edges.json() == []


def test_chat_refuses_price_prediction():
    """Verify POST /chat refuses future price predictions or investment advice."""
    questions = [
        "Will Delta stock price go up after this fuel spike?",
        "Predict the price of oil next month and should I buy XOM shares?",
        "Give me investment advice on Valero stock target price"
    ]
    for q in questions:
        res = client.get("/edges")
        first_edge_id = res.json()[0]["id"]
        chat_res = client.post("/chat", json={"question": q, "edge_id": first_edge_id})
        assert chat_res.status_code == 200
        data = chat_res.json()
        assert data["refusal"] is True
        ans_lower = data["answer"].lower()
        assert any(term in ans_lower for term in ["cannot predict", "investment advice", "refuse", "not predict"])


def test_chat_cites_only_stored_evidence():
    """Verify POST /chat only cites the stored edge quote and source URL."""
    edges_res = client.get("/edges")
    assert edges_res.status_code == 200
    edges = edges_res.json()
    assert len(edges) > 0

    target_edge = edges[0]
    edge_id = target_edge["id"]

    chat_res = client.post("/chat", json={
        "question": "What primary evidence quote supports this relationship?",
        "edge_id": edge_id
    })
    assert chat_res.status_code == 200
    data = chat_res.json()
    assert data["refusal"] is False
    answer = data["answer"]

    # Verify answer explains stored evidence quote
    assert target_edge["evidence_quote"] in answer or target_edge["dst"] in answer
    # Verify primary citation matches stored source URL
    assert target_edge["source_url"] in answer

    # Verify no outside URLs are cited
    import re
    urls = re.findall(r"https?://[^\s)\]]+", answer)
    for u in urls:
        assert u == target_edge["source_url"], f"Answer cited unauthorized external URL: {u}"


def test_chat_uncovered_question():
    """Verify questions outside stored evidence return 'The stored evidence does not say'."""
    edges_res = client.get("/edges")
    edge_id = edges_res.json()[0]["id"]

    chat_res = client.post("/chat", json={
        "question": "What is the CEO's favorite sports team and personal hobby?",
        "edge_id": edge_id
    })
    assert chat_res.status_code == 200
    data = chat_res.json()
    assert "the stored evidence does not say" in data["answer"].lower()


def test_evidence_list_and_node_lookups():
    """Verify GET /evidence returns list and GET /evidence/{edge_id} handles node names and tickers."""
    res_list = client.get("/evidence")
    assert res_list.status_code == 200
    evidence_items = res_list.json()
    assert len(evidence_items) > 0
    assert "evidence_quote" in evidence_items[0]
    assert "source_url" in evidence_items[0]

    # Node ticker lookup: DAL
    res_dal = client.get("/evidence/DAL")
    assert res_dal.status_code == 200
    dal_data = res_dal.json()
    assert dal_data["company"] == "DAL" or dal_data["dst"] == "DAL"
    assert len(dal_data["evidence_quote"]) > 0

    # Sector / Node name lookup: Airlines
    res_air = client.get("/evidence/Airlines")
    assert res_air.status_code == 200
    assert len(res_air.json()["evidence_quote"]) > 0


def test_fingraph_dashboard_served():
    """Verify root GET / serves fingraph-dashboard.html with FinGraph AI."""
    res = client.get("/")
    assert res.status_code == 200
    assert "FinGraph AI" in res.text
    assert "Financial Intelligence Overview" in res.text
    assert "Event Impact Graph" in res.text
    assert "Impact Chain" in res.text


def test_validation_endpoint_and_wording_rule():
    """Verify GET /validation returns real counts and strictly complies with wording rules."""
    res = client.get("/validation")
    assert res.status_code == 200
    data = res.json()

    assert "counts" in data
    assert "consistent" in data["counts"]
    assert "not_consistent" in data["counts"]
    assert "confounded" in data["counts"]
    assert data["counts"]["total"] > 0

    assert "text_summary" in data
    summary = data["text_summary"]
    assert len(summary) > 0

    # Rule 5: Using ONLY "consistent with" or "not consistent with", NEVER "proves"
    assert "consistent with" in summary.lower(), "Summary must contain 'consistent with'"
    assert "proves" not in summary.lower(), "Wording rule violated: 'proves' is forbidden"
    assert "proven" not in summary.lower(), "Wording rule violated: 'proven' is forbidden"
    assert "proof" not in summary.lower(), "Wording rule violated: 'proof' is forbidden"


def test_flow_validation_column_populated():
    """Verify GET /flow validation column reflects empirical validation data."""
    res = client.get("/flow")
    assert res.status_code == 200
    data = res.json()
    val_col = data["columns"][5]
    assert val_col["name"] == "Validation"
    assert len(val_col["nodes"]) > 0

    # Confirm column sum matches total
    col_sum = sum(n["value"] for n in val_col["nodes"])
    assert col_sum == data["total"]


def test_metrics_endpoint_and_confidence_intervals():
    """Verify GET /metrics returns all evaluation metrics with confidence intervals."""
    res = client.get("/metrics")
    assert res.status_code == 200
    data = res.json()

    # Required dimensions
    assert "valid_quote" in data
    assert "edge_precision" in data
    assert "tier_accuracy" in data
    assert "inter_reviewer_agreement" in data
    assert "latency" in data
    assert "leakage_test" in data

    # Confidence interval format
    for metric_key in ["valid_quote", "edge_precision", "tier_accuracy"]:
        m = data[metric_key]
        assert "count" in m
        assert "total" in m
        assert "ci_lower" in m
        assert "ci_upper" in m
        assert "display" in m
        assert m["ci_lower"] <= m["proportion"] <= m["ci_upper"]
        assert "CI:" in m["display"]

    # Inter-reviewer agreement
    agree = data["inter_reviewer_agreement"]
    assert "claim_support" in agree
    assert "kappa" in agree["claim_support"]
    assert "raw_agreement" in agree["claim_support"]

    # Traversal benchmark
    lat = data["latency"]
    assert "median_ms" in lat
    assert "p95_ms" in lat

    # Point-in-time leakage audit
    leak = data["leakage_test"]
    assert "event_date" in leak
    assert "post_event_edges_dropped" in leak
    assert "leakage_score_l1_norm" in leak


def test_dashboard_metrics_and_limitations_panels():
    """Verify dashboard HTML contains Metrics section and Limitations panel."""
    res = client.get("/")
    assert res.status_code == 200
    html = res.text

    # Metrics section
    assert "metrics-card" in html or "Evaluation Metrics" in html
    assert "metric-valid-quote" in html
    assert "metric-edge-precision" in html

    # Limitations panel
    assert "limitations-card" in html or "System Limitations" in html
    assert "Small Sample" in html or "small sample" in html.lower()
    assert "US Large Caps Only" in html or "us large caps" in html.lower()
    assert "Supply-Chain Data Incomplete" in html or "supply-chain data incomplete" in html.lower()
    assert "Not Investment Advice" in html or "not investment advice" in html.lower()



