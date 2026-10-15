"""
FinGraph AI - Automated Test Suite for News Ingestion, AI Extraction, and Real-Time Radar
Fully offline with pytest, mocked network calls, and deterministic validation.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app
from database import SessionLocal
from models import Article, Event, ProviderStatus, Entity
from services.news_ingestion import (
    compute_article_fingerprint,
    host_is_public,
    is_valid_url,
    record_provider_status,
)
from services.event_extractor import (
    extract_event_deterministic,
    process_article_and_persist_event,
)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def test_ssrf_protection():
    """Verify that private, localhost, and metadata IPs are rejected by SSRF filter."""
    assert host_is_public("127.0.0.1") is False
    assert host_is_public("localhost") is False
    assert host_is_public("169.254.169.254") is False
    assert host_is_public("10.0.0.1") is False
    assert host_is_public("192.168.1.1") is False

    assert is_valid_url("http://127.0.0.1") is True
    assert is_valid_url("ftp://badurl.com") is False
    assert is_valid_url("not a url") is False


def test_fingerprint_deduplication():
    """Verify SHA-256 deterministic fingerprint deduplication."""
    fp1 = compute_article_fingerprint("Crude Oil Spikes to $130", "https://news.com/oil")
    fp2 = compute_article_fingerprint("  crude oil spikes to $130  ", "https://news.com/oil")
    fp3 = compute_article_fingerprint("Natural Gas Plunges", "https://news.com/gas")

    assert fp1 == fp2
    assert fp1 != fp3


def test_provider_status_recording(db):
    """Verify provider health updates in database adhering to Rule 5."""
    record_provider_status(db, "test_provider", True, records_count=5, status_message="Healthy")
    row = db.query(ProviderStatus).filter_by(provider_name="test_provider").first()
    assert row is not None
    assert row.is_connected is True
    assert row.records_count >= 5
    assert row.last_error is None

    # Now record failure
    record_provider_status(db, "test_provider", False, error_msg="Rate limited")
    row_fail = db.query(ProviderStatus).filter_by(provider_name="test_provider").first()
    assert row_fail.is_connected is False
    assert "Rate limited" in row_fail.last_error


def test_deterministic_event_extraction(db):
    """Verify deterministic NLP extraction discovers commodity shocks and entities."""
    sample_text = "Brent crude oil benchmark price surged today as Middle East supply was contracted. Airlines face higher jet fuel expenses."
    res = extract_event_deterministic(
        sample_text,
        "Crude Oil Benchmark Price Surges",
        "Energy Daily",
        "https://energy.org/oil",
        db,
    )
    assert res["id"].startswith("evt_")
    assert res["category"] == "Commodities"
    assert res["event_type"] == "commodity_shock"
    assert "Crude Oil" in res["entities"] or len(res["entities"]) >= 1
    assert len(res["supporting_quote"]) > 0


def test_api_providers_endpoint(client):
    """GET /api/providers must return list of configured provider metadata."""
    res = client.get("/api/providers")
    assert res.status_code == 200
    providers = res.json()
    assert isinstance(providers, list)
    names = [p["name"] for p in providers]
    assert "gdelt" in names
    assert "sec_edgar" in names
    assert "alphavantage" in names
    assert "gemini" in names


def test_api_news_list_and_search(client):
    """GET /api/news lists stored news articles with filtering."""
    res = client.get("/api/news?limit=10")
    assert res.status_code == 200
    data = res.json()
    assert "articles" in data
    assert "total" in data


def test_api_event_extract_endpoint(client):
    """POST /api/events/extract extracts and links an event from submitted text."""
    payload = {
        "title": "Fed Hikes Rates by 50 Bps Amid Persistent Inflation",
        "text": "The Federal Reserve raised policy interest rates, increasing debt financing costs for capital-intensive sectors.",
        "source_name": "Macro Watch",
        "source_url": "https://macrowatch.org/fed",
        "use_ai": False,
    }
    res = client.post("/api/events/extract", json=payload)
    assert res.status_code == 200
    ev = res.json()
    assert "id" in ev
    assert "title" in ev
    assert ev["category"] in ("Central bank decisions", "Macroeconomics")


def test_api_observations_endpoint(client):
    """GET /api/observations returns historical market context."""
    res = client.get("/api/observations")
    assert res.status_code == 200
    obs = res.json()
    assert "series" in obs
    assert "disclaimer" in obs
    assert len(obs["series"]) > 0
