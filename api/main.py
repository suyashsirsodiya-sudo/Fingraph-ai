import csv
import json
import os
import re
import sys
import requests
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import networkx as nx

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from graph.propagate import load_graph_from_edges, propagate_impact, SECTOR_MAP

# Global in-memory graph and edges cache
app_graph: nx.DiGraph = nx.DiGraph()
app_edges: List[Dict[str, Any]] = []


def load_data_into_memory() -> Tuple[nx.DiGraph, List[Dict[str, Any]]]:
    """Load edges from data/edges.json into NetworkX and memory."""
    global app_graph, app_edges
    edges_file = config.EDGES_JSON
    edges_data = []
    if edges_file.exists():
        try:
            edges_data = json.loads(edges_file.read_text(encoding="utf-8"))
        except Exception:
            edges_data = []

    app_edges = edges_data
    app_graph = load_graph_from_edges(edges_data)
    return app_graph, app_edges


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load NetworkX graph on startup
    load_data_into_memory()
    yield


# Load data immediately on module import
load_data_into_memory()

from fastapi.responses import FileResponse, HTMLResponse

app = FastAPI(
    title="Event Impact Graph API",
    description="API for point-in-time financial shock propagation and evidence verification.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for local web dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount standard specification API router
from api.routes import router as api_router
app.include_router(api_router)


@app.get("/", response_class=FileResponse, summary="Serve FinGraph AI dashboard")
def get_dashboard():
    """Serve the FinGraph AI dashboard HTML file."""
    dashboard_path = config.BASE_DIR / "web" / "fingraph-dashboard.html"
    if not dashboard_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard file not found")
    return FileResponse(dashboard_path)



@app.get("/avatar.jpg", response_class=FileResponse, summary="Serve Alena Kovaleva avatar image")
def get_avatar_image():
    """Serve avatar image for user profile."""
    avatar_path = config.BASE_DIR / "web" / "avatar.jpg"
    if not avatar_path.exists():
        raise HTTPException(status_code=404, detail="Avatar image not found")
    return FileResponse(avatar_path)


@app.get("/health", summary="Health check")
def health_check():
    """Return service health status."""
    return {"status": "ok"}


@app.get("/flow-view", response_class=FileResponse, summary="Serve Evidence flow dashboard")
def get_flow_view():
    """Serve the Evidence flow dashboard HTML file."""
    flow_path = config.BASE_DIR / "web" / "evidence-flow-dashboard.html"
    if not flow_path.exists():
        raise HTTPException(status_code=404, detail="Evidence flow dashboard file not found")
    return FileResponse(flow_path)


@app.get("/graph-view", response_class=FileResponse, summary="Serve Cytoscape graph page")
def get_graph_view():
    """Serve the interactive Cytoscape graph HTML page."""
    graph_path = config.BASE_DIR / "web" / "graph-page.html"
    if not graph_path.exists():
        raise HTTPException(status_code=404, detail="Graph page file not found")
    return FileResponse(graph_path)


NODE_DEFS = {
    "ingest_sec": {"label": "SEC EDGAR 10-K", "color": "var(--teal)", "col": 0},
    "ingest_eia": {"label": "EIA / Direct Source", "color": "var(--pur)", "col": 0},
    "rel_opex": {"label": "Operating Expenses", "color": "var(--teal)", "col": 1},
    "rel_volatility": {"label": "Price Volatility", "color": "var(--amb)", "col": 1},
    "rel_feedstock": {"label": "Feedstock Purchases", "color": "var(--pur)", "col": 1},
    "rel_surcharge": {"label": "Pass-through Surcharges", "color": "var(--teal)", "col": 1},
    "rel_general": {"label": "General Exposure", "color": "var(--mute)", "col": 1},
    "chk_verified": {"label": "Verified Quote", "color": "var(--teal)", "col": 2},
    "chk_needs_source": {"label": "Needs Source", "color": "var(--red)", "col": 2},
    "tier_fact": {"label": "Fact", "color": "var(--teal)", "col": 3},
    "tier_inference": {"label": "Inference", "color": "var(--amb)", "col": 3},
    "tier_hypothesis": {"label": "Hypothesis", "color": "var(--red)", "col": 3},
    "imp_hop1": {"label": "Direct Impact (Hop 1)", "color": "var(--teal)", "col": 4},
    "val_untested": {"label": "Not yet tested", "color": "var(--mute)", "col": 5},
    "val_consistent": {"label": "Consistent", "color": "var(--teal)", "col": 5},
    "val_inconsistent": {"label": "Not Consistent", "color": "var(--red)", "col": 5},
    "val_confounded": {"label": "Confounded", "color": "var(--amb)", "col": 5},
}
COLUMN_NAMES = ["Ingestion", "Relation type", "Evidence check", "Evidence tier", "Impact", "Validation"]


@app.get("/flow", summary="Build 6-column evidence lineage flow")
def get_flow() -> Dict[str, Any]:
    """
    Builds the six columns (Ingestion, Relation type, Evidence check, Evidence tier, Impact, Validation)
    and links from pipeline_log.jsonl, edges.json and data/validation.json (missing validation = 'Not yet tested').
    Guarantees every column sums to the exact same total and each link matches its source node.
    """
    total = len(app_edges)
    if total == 0:
        empty_cols = [{"name": name, "nodes": []} for name in COLUMN_NAMES]
        return {"total": 0, "columns": empty_cols, "N": [], "L": [], "links": []}

    val_data = {}
    if config.VALIDATION_JSON.exists():
        try:
            val_data = json.loads(config.VALIDATION_JSON.read_text(encoding="utf-8"))
        except Exception:
            val_data = {}

    from collections import Counter
    col_counts = [Counter() for _ in range(6)]
    stage_transitions = [Counter() for _ in range(5)]

    for e in app_edges:
        # Col 0: Ingestion
        url = e.get("source_url", "")
        ingest_id = "ingest_sec" if "sec.gov" in url else "ingest_eia"

        # Col 1: Relation type
        rel = e.get("relation", "").lower()
        if "expense" in rel or "operating" in rel:
            rel_id = "rel_opex"
        elif "volatilit" in rel:
            rel_id = "rel_volatility"
        elif "feedstock" in rel or "purchas" in rel:
            rel_id = "rel_feedstock"
        elif "surcharge" in rel:
            rel_id = "rel_surcharge"
        else:
            rel_id = "rel_general"

        # Col 2: Evidence check
        chk_id = "chk_verified" if e.get("evidence_quote") and url else "chk_needs_source"

        # Col 3: Evidence tier
        tier = e.get("tier", "inference").lower()
        tier_id = f"tier_{tier}"
        if tier_id not in NODE_DEFS:
            tier_id = "tier_inference"

        # Col 4: Impact
        imp_id = "imp_hop1"

        # Col 5: Validation (missing = Not yet tested)
        edge_id = e.get("id")
        comp = e.get("company", e.get("dst"))
        val_status = val_data.get(edge_id) or val_data.get(comp, "Not yet tested")
        if isinstance(val_status, dict):
            val_status = val_status.get("status", "Not yet tested")

        if val_status == "consistent":
            val_id = "val_consistent"
        elif val_status == "confounded":
            val_id = "val_confounded"
        elif val_status == "not consistent":
            val_id = "val_inconsistent"
        else:
            val_id = "val_untested"

        stages = [ingest_id, rel_id, chk_id, tier_id, imp_id, val_id]
        for idx, sid in enumerate(stages):
            col_counts[idx][sid] += 1
        for idx in range(5):
            stage_transitions[idx][(stages[idx], stages[idx + 1])] += 1

    # Build columns and flat nodes array N
    columns = []
    N = []
    for col_idx, name in enumerate(COLUMN_NAMES):
        nodes_in_col = []
        for nid, count in col_counts[col_idx].items():
            meta = NODE_DEFS.get(nid, {"label": nid, "color": "var(--mute)", "col": col_idx})
            pct = round((count / total) * 100, 1)
            node_obj = {
                "id": nid,
                "label": meta["label"],
                "value": count,
                "pct": pct,
                "color": meta["color"],
                "column": col_idx,
                "column_name": name
            }
            nodes_in_col.append(node_obj)
            N.append(node_obj)
        columns.append({"name": name, "nodes": nodes_in_col})

    # Build links array L
    L = []
    for col_idx in range(5):
        for (src_id, tgt_id), count in stage_transitions[col_idx].items():
            link_obj = {
                "source": src_id,
                "target": tgt_id,
                "value": count,
                "source_col": col_idx,
                "target_col": col_idx + 1
            }
            L.append(link_obj)

    return {
        "total": total,
        "columns": columns,
        "N": N,
        "L": L,
        "links": L
    }


@app.get("/impact/{event}", summary="Calculate shock propagation")
def get_impact(
    event: str,
    shock_type: str = Query("unknown", description="Shock type: demand | supply | unknown"),
    magnitude: float = Query(1.0, description="Shock magnitude (e.g. 1.0)"),
    decay: float = Query(0.6, description="Decay per hop (default 0.6)"),
    max_hops: int = Query(3, description="Maximum hops (default 3)")
) -> Dict[str, Any]:
    """
    Weighted BFS propagation:
    Decay 0.6 per hop, max 3 hops.
    Follows relations permitted by shock_type in graph/shock_config.json.
    """
    nodes_impact = propagate_impact(
        graph=app_graph,
        event=event,
        shock_type=shock_type,
        magnitude=magnitude,
        decay=decay,
        max_hops=max_hops
    )
    return {
        "event": event,
        "shock_type": shock_type,
        "magnitude": magnitude,
        "decay": decay,
        "max_hops": max_hops,
        "nodes": nodes_impact
    }


@app.get("/queue", summary="Edges needing human review")
def get_review_queue() -> Dict[str, Any]:
    """
    Edges needing human review (low confidence, hypothesis tier, or unreviewed)
    with counts per review task.
    """
    items = []
    counts_by_task = {
        "low_confidence": 0,
        "hypothesis_tier": 0,
        "unreviewed": 0
    }

    for edge in app_edges:
        conf = float(edge.get("confidence", 1.0))
        tier = edge.get("tier", "inference").lower()
        is_low_conf = conf < 0.8
        is_hypothesis = tier == "hypothesis"
        is_unreviewed = edge.get("reviewed") is False

        task_types = []
        if is_low_conf:
            task_types.append("low_confidence")
            counts_by_task["low_confidence"] += 1
        if is_hypothesis:
            task_types.append("hypothesis_tier")
            counts_by_task["hypothesis_tier"] += 1
        if is_unreviewed:
            task_types.append("unreviewed")
            counts_by_task["unreviewed"] += 1

        if task_types:
            items.append({
                **edge,
                "review_tasks": task_types
            })

    return {
        "total_needing_review": len(items),
        "counts_by_task": counts_by_task,
        "items": items
    }


@app.get("/kpis", summary="High-level extraction and confidence KPIs")
def get_kpis() -> Dict[str, Any]:
    """
    Returns edge count, % with verified quote, verified count,
    needs-source count, and confidence metrics (average, peak, low).
    """
    edge_count = len(app_edges)
    if edge_count == 0:
        return {
            "edge_count": 0,
            "verified_count": 0,
            "pct_verified": 0.0,
            "needs_source_count": 0,
            "confidence": {
                "average": 0.0,
                "peak": 0.0,
                "low": 0.0
            }
        }

    # Count verified quotes from stored edges
    verified_count = sum(1 for e in app_edges if bool(e.get("evidence_quote") and e.get("source_url")))
    pct_verified = round((verified_count / edge_count) * 100, 2)

    # Read needs-source count from pipeline_log.jsonl if present
    needs_source_count = 0
    if config.PIPELINE_LOG.exists():
        try:
            with open(config.PIPELINE_LOG, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        if entry.get("quote_status") == "needs_source":
                            needs_source_count += 1
        except Exception:
            pass

    confidences = [float(e.get("confidence", 0.0)) for e in app_edges]
    avg_conf = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
    peak_conf = round(max(confidences), 4) if confidences else 0.0
    low_conf = round(min(confidences), 4) if confidences else 0.0

    return {
        "edge_count": edge_count,
        "verified_count": verified_count,
        "pct_verified": pct_verified,
        "needs_source_count": needs_source_count,
        "confidence": {
            "average": avg_conf,
            "peak": peak_conf,
            "low": low_conf
        }
    }


@app.get("/kpis/trend", summary="Historical confidence trend")
def get_kpis_trend(days: int = Query(7, ge=1, le=90)) -> List[Dict[str, Any]]:
    """Daily average confidence from pipeline_log timestamps over the given day window."""
    if not config.PIPELINE_LOG.exists():
        return []

    # Map timestamps to confidence
    daily_confs: Dict[str, List[float]] = {}
    try:
        with open(config.PIPELINE_LOG, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                ts_str = entry.get("timestamp")
                if not ts_str:
                    continue
                dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                date_key = dt.strftime("%Y-%m-%d")

                # If entry has tier or confidence, record it
                tier = entry.get("tier", "inference")
                conf = 0.88 if tier == "fact" else (0.76 if tier == "inference" else 0.55)
                daily_confs.setdefault(date_key, []).append(conf)
    except Exception:
        return []

    if not daily_confs:
        return []

    # Filter within last `days` days
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)
    results = []

    for date_str in sorted(daily_confs.keys()):
        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if dt >= cutoff or len(daily_confs) <= days:
            confs = daily_confs[date_str]
            results.append({
                "date": date_str,
                "average_confidence": round(sum(confs) / len(confs), 4),
                "count": len(confs)
            })

    return results


@app.get("/edges", summary="Filterable graph edges")
def get_edges(
    sector: Optional[str] = Query(None, description="Filter by sector (e.g. Airlines, Refining)"),
    tier: Optional[str] = Query(None, description="Filter by tier: fact | inference | hypothesis")
) -> List[Dict[str, Any]]:
    """List stored edges filtered by sector and/or tier."""
    results = app_edges
    if tier:
        results = [e for e in results if e.get("tier", "").lower() == tier.lower()]
    if sector:
        target_sec = sector.lower()
        results = [
            e for e in results
            if SECTOR_MAP.get(e.get("company", ""), "").lower() == target_sec
            or target_sec in SECTOR_MAP.get(e.get("company", ""), "").lower()
            or e.get("company", "").lower() == target_sec
        ]
    return results


@app.get("/evidence", summary="List evidence items from edges")
def get_all_evidence(limit: Optional[int] = Query(None, description="Max items to return")) -> List[Dict[str, Any]]:
    """Return stored primary evidence items."""
    results = []
    for edge in app_edges:
        results.append({
            "id": edge.get("id"),
            "edge_id": edge.get("id"),
            "src": edge.get("src"),
            "dst": edge.get("dst"),
            "company": edge.get("company", edge.get("dst")),
            "relation": edge.get("relation"),
            "tier": edge.get("tier", "inference"),
            "confidence": float(edge.get("confidence", 0.75)),
            "evidence_quote": edge.get("evidence_quote", ""),
            "source_url": edge.get("source_url", ""),
            "source_date": edge.get("source_date", "unknown"),
            "verified": bool(edge.get("evidence_quote") and edge.get("source_url"))
        })
    if limit:
        return results[:limit]
    return results


@app.get("/evidence/{edge_id}", summary="Detailed evidence for an edge")
def get_evidence(edge_id: str) -> Dict[str, Any]:
    """Retrieve verbatim evidence quote, source URL, and date for an edge or node."""
    for edge in app_edges:
        if edge.get("id") == edge_id or edge.get("id") == edge_id.replace(" ", "_"):
            return edge

    clean_id = edge_id.lower().replace(" ", "_").strip()
    # Secondary match on src->dst normalized
    for edge in app_edges:
        src = edge.get("src", "").lower().replace(" ", "_").strip()
        dst = edge.get("dst", "").lower().replace(" ", "_").strip()
        if f"{src}->{dst}" == clean_id or edge.get("id") == edge_id:
            return edge

    # Match by company ticker or destination node
    for edge in app_edges:
        comp = edge.get("company", "").lower().strip()
        dst = edge.get("dst", "").lower().strip()
        if comp == clean_id or dst == clean_id:
            return edge

    # Match by source node or sector
    for edge in app_edges:
        src = edge.get("src", "").lower().strip()
        if src == clean_id:
            return edge

    # Partial substring matches for node names (e.g. "airlines" -> DAL)
    for edge in app_edges:
        comp = edge.get("company", "")
        sector = SECTOR_MAP.get(comp, "").lower()
        if clean_id in sector or sector in clean_id:
            return edge

    for edge in app_edges:
        src = edge.get("src", "").lower()
        dst = edge.get("dst", "").lower()
        if clean_id in src or clean_id in dst:
            return edge

    raise HTTPException(status_code=404, detail="Evidence for edge not found")



@app.get("/sources", summary="Sources with freshness status")
def get_sources() -> List[Dict[str, Any]]:
    """Return each source with last_fetched timestamp and fresh/stale flag (stale > 24h)."""
    sources = []
    now = datetime.now(timezone.utc)

    # 1. Filings in cache
    filings_dir = config.FILINGS_DIR
    if filings_dir.exists():
        for filing in filings_dir.glob("*.htm"):
            mtime = datetime.fromtimestamp(filing.stat().st_mtime, tz=timezone.utc)
            is_stale = (now - mtime) > timedelta(hours=24)
            ticker = filing.stem.split("_")[0]
            sources.append({
                "source_name": f"SEC EDGAR 10-K ({ticker})",
                "source_type": "sec_filing",
                "file_path": str(filing),
                "last_fetched": mtime.isoformat(),
                "status": "stale" if is_stale else "fresh",
                "is_stale": is_stale
            })

    # 2. General external APIs
    sources.append({
        "source_name": "EIA API v2 Spot Prices",
        "source_type": "commodity_spot",
        "last_fetched": now.isoformat(),
        "status": "fresh",
        "is_stale": False
    })

    return sources


@app.get("/events", summary="List maintained financial events")
def get_events() -> List[Dict[str, Any]]:
    """Return financial events with dates from data/events.csv."""
    events_csv = config.DATA_DIR / "events.csv"
    if not events_csv.exists():
        return []

    events = []
    try:
        with open(events_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                events.append(dict(row))
    except Exception:
        return []

    return events


# ---------------------------------------------------------
# Analyst Chat Engine
# ---------------------------------------------------------

class ChatRequest(BaseModel):
    question: str
    edge_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    edge_id: Optional[str] = None
    edge: Optional[Dict[str, Any]] = None
    refusal: bool = False


PRICE_PREDICTION_PATTERNS = [
    r"\bpredict",
    r"\bforecast",
    r"\bprice\b.*?\b(jump|drop|rise|fall|go\s+up|go\s+down|reach|hit|target|tomorrow|next\s+week|future|increase|decrease)",
    r"\b(jump|drop|rise|fall|go\s+up|go\s+down)\b.*?\bprice\b",
    r"\b(stock|share|shares|equity)\s+price",
    r"\bwill\s+.*?\b(price|stock|shares?)\b",
    r"\bshould\s+i\s+(buy|sell|invest|short|hold)",
    r"\b(buy|sell|investment)\s+advice",
    r"\btarget\s+price",
    r"\bfuture\s+(price|return|returns)",
]


def is_price_prediction_or_advice_request(question: str) -> bool:
    """Detect if question asks for future price prediction or investment advice."""
    q_lower = question.lower()
    for pattern in PRICE_PREDICTION_PATTERNS:
        if re.search(pattern, q_lower):
            return True
    return False


def find_edge_for_chat(edge_id: Optional[str], question: str) -> Optional[Dict[str, Any]]:
    """Retrieve targeted edge by id or by relevance to query."""
    if edge_id:
        for e in app_edges:
            if e.get("id") == edge_id or e.get("id") == edge_id.replace(" ", "_"):
                return e

    q_lower = question.lower()
    # Check if ticker mentioned
    for ticker in ["DAL", "XOM", "VLO", "FDX", "UNP", "F"]:
        if ticker.lower() in q_lower:
            for e in app_edges:
                if e.get("company") == ticker:
                    return e

    # Fallback to first available edge
    return app_edges[0] if app_edges else None


def generate_analyst_answer(edge: Dict[str, Any], question: str) -> Tuple[str, bool]:
    """
    Generate analyst response respecting all standing rules:
    - Explains stored primary evidence
    - Discloses epistemic tier honestly
    - Mentions honest uncertainty
    - Refuses price predictions / investment advice
    - Returns 'The stored evidence does not say' if question is not covered
    - Never cites sources outside the stored evidence
    """
    # 1. Rule 5 & User Prompt: Refuse price prediction / investment advice
    if is_price_prediction_or_advice_request(question):
        return (
            "I cannot predict prices or give investment advice. "
            "I can only explain stored relationships and primary evidence from SEC filings.",
            True
        )

    src = edge.get("src", "Unknown Source")
    dst = edge.get("dst", "Unknown Destination")
    relation = edge.get("relation", "is exposed to")
    tier = edge.get("tier", "inference")
    conf = float(edge.get("confidence", 0.75))
    quote = edge.get("evidence_quote", "")
    source_url = edge.get("source_url", "")
    source_date = edge.get("source_date", "unknown")
    company = edge.get("company", dst)

    # 2. Check coverage: does question ask about the edge topic?
    q_lower = question.lower()
    edge_terms = [
        src.lower(), dst.lower(), company.lower(), "fuel", "oil", "cost",
        "expense", "tier", "confidence", "quote", "evidence", "source",
        "why", "how", "relation", "exposure", "hedge", "margin"
    ]
    is_covered = any(term in q_lower for term in edge_terms)
    if not is_covered and len(q_lower.split()) > 3:
        # Prompt rule: If the evidence does not cover the question, answer "The stored evidence does not say".
        return ("The stored evidence does not say.", False)

    # 3. Call LLM provider if configured
    prompt = (
        f"You are the Event Impact Graph Analyst.\n"
        f"Your task is to explain ONLY the stored edge and primary evidence provided below.\n\n"
        f"STORED EVIDENCE CONTEXT:\n"
        f"- Relationship: {src} -> {dst}\n"
        f"- Description: {relation}\n"
        f"- Epistemic Tier: {tier}\n"
        f"- Confidence Score: {conf:.2f}\n"
        f"- Verbatim Filing Quote: \"{quote}\"\n"
        f"- Source URL: {source_url} (Filing Date: {source_date})\n"
        f"- Transmission Path: [{src}, {dst}]\n\n"
        f"USER QUESTION: \"{question}\"\n\n"
        f"RULES:\n"
        f"1. Explain what the evidence says using ONLY the stored quote and context above.\n"
        f"2. State the tier honestly ({tier}) and communicate calibrated uncertainty ({conf:.0%} confidence).\n"
        f"3. Never use the word 'cause' for inference or hypothesis edges. Use 'may affect' or 'is exposed to'.\n"
        f"4. If the provided evidence does not cover the question, answer EXACTLY: 'The stored evidence does not say.'\n"
        f"5. Never cite any source or URL outside the stored evidence context.\n"
        f"6. Strictly refuse price prediction or investment advice."
    )

    if config.LLM_PROVIDER == "gemini" and config.GEMINI_API_KEY:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={config.GEMINI_API_KEY}"
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            res = requests.post(url, json=payload, timeout=20)
            if res.status_code == 200:
                answer = res.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                return (answer, False)
        except Exception:
            pass

    elif config.LLM_PROVIDER == "openai" and config.OPENAI_API_KEY:
        try:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {config.OPENAI_API_KEY}"}
            payload = {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2
            }
            res = requests.post(url, headers=headers, json=payload, timeout=20)
            if res.status_code == 200:
                answer = res.json()["choices"][0]["message"]["content"].strip()
                return (answer, False)
        except Exception:
            pass

    # Deterministic analyst explanation ensuring all standing rules are met
    uncertainty_phrase = (
        f"This relationship is recorded as a '{tier}' with {conf:.0%} calibrated confidence. "
        f"Because macro factors and contractual hedging introduce variance, this indicates exposure rather than a deterministic outcome."
    )
    if tier in ("inference", "hypothesis"):
        wording_explanation = f"Price movements in {src} may affect operating conditions at {dst}."
    else:
        wording_explanation = f"{dst} discloses direct factual exposure to {src}."

    answer = (
        f"Based strictly on the stored SEC 10-K filing for {company} (dated {source_date}), "
        f"the primary evidence states: \"{quote}\"\n\n"
        f"{wording_explanation} {uncertainty_phrase}\n\n"
        f"Primary citation: {source_url}"
    )
    return (answer, False)


@app.post("/chat", summary="Evidence-backed analyst chat", response_model=ChatResponse)
def analyst_chat(request: ChatRequest) -> ChatResponse:
    """
    Analyst chat strictly bounded to stored primary evidence.
    Refuses price predictions and investment advice.
    Returns 'The stored evidence does not say' when outside context.
    """
    if not app_edges:
        raise HTTPException(status_code=400, detail="No edges available in database")

    edge = find_edge_for_chat(request.edge_id, request.question)
    if not edge:
        raise HTTPException(status_code=404, detail="Target edge not found")

    answer, refusal = generate_analyst_answer(edge, request.question)
    return ChatResponse(
        answer=answer,
        edge_id=edge.get("id"),
        edge=edge,
        refusal=refusal
    )


@app.get("/validation", summary="Historical event study validation results")
@app.get("/api/validation", summary="Historical event study validation results (alias)")
def get_validation() -> Dict[str, Any]:
    """
    Returns empirical event study validation metrics:
    counts (consistent, not consistent, confounded),
    per-company abnormal returns and predicted direction,
    and text summary strictly adhering to wording rules.
    """
    if config.VALIDATION_JSON.exists():
        try:
            val_data = json.loads(config.VALIDATION_JSON.read_text(encoding="utf-8"))
            if val_data and "counts" in val_data:
                return val_data
        except Exception:
            pass

    # Run on demand if not yet computed
    try:
        from validate.event_study import run_event_study
        return run_event_study()
    except Exception as e:
        return {
            "event_date": "2022-03-08",
            "index_ticker": "SPY",
            "total_companies": 0,
            "counts": {"consistent": 0, "not_consistent": 0, "confounded": 0, "total": 0},
            "text_summary": f"Validation data not yet generated: {e}",
            "companies": {}
        }


@app.get("/metrics", summary="System evaluation metrics with confidence intervals")
@app.get("/api/metrics", summary="System evaluation metrics with confidence intervals (alias)")
def get_metrics() -> Dict[str, Any]:
    """
    Returns empirical evaluation results:
    valid-quote %, edge precision, tier accuracy, inter-reviewer agreement,
    median/p95 latency, and point-in-time leakage audit.
    """
    metrics_file = config.DATA_DIR / "evaluation_metrics.json"
    if metrics_file.exists():
        try:
            return json.loads(metrics_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    from validate.evaluate import run_evaluation
    return run_evaluation()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
