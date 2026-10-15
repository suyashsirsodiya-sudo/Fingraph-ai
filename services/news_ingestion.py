"""
FinGraph AI - Real-Time Financial News Ingestion & Provider Health System
Implements permitted, configurable news discovery:
1. GDELT Project (global news discovery, live event queries)
2. Alpha Vantage (financial news & sentiment)
3. SEC EDGAR (official corporate filings and submissions)

Security & Reliability:
- Server-side secret management
- SSRF protection (host_is_public, blocks private IPs / localhost)
- Request timeouts & response size limits
- Fingerprint deduplication (SHA-256)
- Provider health tracking (only 'Connected' when tested in last 24h)
"""

import hashlib
import ipaddress
import json
import os
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

from sqlalchemy.orm import Session
from database import SessionLocal
from models import Article, ProviderStatus, utc_now
import config

UA = "FinGraphAI/1.0 (EventImpactGraphResearch contact@hackathon.org)"


# ── SSRF Protection & Safe Network Client ─────────────────────────────────

def is_valid_url(url: str) -> bool:
    try:
        p = urlparse(url)
        return p.scheme in ("http", "https") and bool(p.hostname) and len(url) <= 2048
    except Exception:
        return False


def host_is_public(host: str) -> bool:
    """True only if resolved IP is globally routable (blocks LAN, localhost, metadata 169.254.169.254)."""
    try:
        infos = socket.getaddrinfo(host, None)
        if not infos:
            return False
        for info in infos:
            ip_str = info[4][0]
            ip_obj = ipaddress.ip_address(ip_str)
            if not ip_obj.is_global:
                return False
        return True
    except Exception:
        return False


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


_opener = urllib.request.build_opener(_NoRedirectHandler)


def safe_fetch_json(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 12,
    max_bytes: int = 2_000_000,
) -> Tuple[int, Any]:
    """Safe HTTP GET with SSRF protection, size caps, and JSON decoding."""
    full_url = url
    if params:
        encoded = urllib.parse.urlencode({k: str(v) for k, v in params.items()})
        full_url = f"{url}?{encoded}" if "?" not in url else f"{url}&{encoded}"

    if not is_valid_url(full_url):
        raise ValueError(f"Invalid or disallowed URL scheme: {url}")

    hostname = urlparse(full_url).hostname
    if not hostname or not host_is_public(hostname):
        raise ValueError(f"SSRF Protection: Host '{hostname}' is not a permitted public routable address")

    req_headers = {"User-Agent": config.SEC_USER_AGENT or UA, "Accept": "application/json"}
    if headers:
        req_headers.update(headers)

    req = urllib.request.Request(full_url, headers=req_headers)
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        code = resp.getcode()
        raw_bytes = resp.read(max_bytes)
        charset = resp.headers.get_content_charset() or "utf-8"
        text = raw_bytes.decode(charset, errors="replace")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = {"raw": text[:500]}
        return code, parsed
    except urllib.error.HTTPError as e:
        return e.code, {"error": f"HTTP {e.code}"}
    except Exception as e:
        raise RuntimeError(f"Network error contacting {hostname}: {str(e)[:200]}")


def compute_article_fingerprint(title: str, url: Optional[str] = None) -> str:
    """Generate deterministic SHA-256 fingerprint for deduplication."""
    clean_title = (title or "").strip().lower()
    clean_url = (url or "").strip().lower()
    raw = f"{clean_title}::{clean_url}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ── Provider Health Tracking ──────────────────────────────────────────────

def record_provider_status(
    db: Session,
    provider_name: str,
    is_connected: bool,
    error_msg: Optional[str] = None,
    records_count: int = 0,
    status_message: Optional[str] = None,
):
    """Update provider_status row adhering to honesty rules."""
    now = utc_now()
    row = db.query(ProviderStatus).filter_by(provider_name=provider_name).first()
    if not row:
        row = ProviderStatus(provider_name=provider_name)
        db.add(row)

    row.last_tested_at = now
    row.last_attempt_at = now
    row.is_connected = is_connected

    if is_connected:
        row.last_success_at = now
        row.last_error = None
        row.status_message = status_message or f"Connected and verified healthy at {now.strftime('%H:%M:%S UTC')}"
        if records_count > 0:
            row.records_count = (row.records_count or 0) + records_count
    else:
        row.last_error = (error_msg or "Connection check failed")[:500]
        row.status_message = (error_msg or "Failed to connect to provider")[:250]

    db.commit()


# ── Provider 1: GDELT Project Ingestion ───────────────────────────────────

def sync_gdelt_news(
    db: Session,
    query: str = "crude oil OR energy supply OR interest rates",
    max_records: int = 15,
) -> Dict[str, Any]:
    """
    Query GDELT 2.0 Doc API for global financial/economic news.
    Public and free API, does not require an API key.
    """
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": min(max(max_records, 1), 50),
        "format": "json",
    }

    try:
        code, data = safe_fetch_json(url, params=params, timeout=15)
        if code != 200 or not isinstance(data, dict):
            err = f"GDELT returned HTTP {code}"
            record_provider_status(db, "gdelt", False, err)
            return {"provider": "gdelt", "success": False, "error": err, "fetched": 0, "inserted": 0}

        articles_raw = data.get("articles", [])
        inserted_count = 0

        for a in articles_raw:
            a_url = a.get("url")
            a_title = a.get("title") or "Financial Market Event"
            a_domain = a.get("domain") or a.get("sourcecountry") or "GDELT Wire"
            seendate_str = a.get("seendate")  # e.g. 20261011T120000Z

            if not a_url or not is_valid_url(a_url):
                continue

            fp = compute_article_fingerprint(a_title, a_url)
            # Deduplication check
            exists = db.query(Article).filter(
                (Article.fingerprint == fp) | (Article.source_url == a_url)
            ).first()
            if exists:
                continue

            # Parse publication timestamp
            pub_dt = None
            if seendate_str:
                try:
                    pub_dt = datetime.strptime(seendate_str, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
                except ValueError:
                    pub_dt = utc_now()
            else:
                pub_dt = utc_now()

            # Classify category heuristically
            t_lower = a_title.lower()
            cat = "Macroeconomics"
            if any(w in t_lower for w in ["oil", "crude", "energy", "fuel", "gas", "pipeline"]):
                cat = "Commodities"
            elif any(w in t_lower for w in ["airline", "delta", "fedex", "earnings", "revenue", "corp"]):
                cat = "Company news"
            elif any(w in t_lower for w in ["war", "hormuz", "sanction", "geopolitic"]):
                cat = "Geopolitics"

            art_id = f"art_gdelt_{fp[:12]}"
            new_art = Article(
                id=art_id,
                title=a_title[:512],
                source_name=a_domain[:256],
                source_url=a_url[:1024],
                published_at=pub_dt,
                retrieved_at=utc_now(),
                excerpt=f"Retrieved via GDELT Project global news index for query '{query}'.",
                fingerprint=fp,
                category=cat,
                entities=[],
                retrieval_status="Source Retrieved",
                review_status="Pending",
                is_demo=False,
            )
            db.add(new_art)
            inserted_count += 1

        db.commit()
        record_provider_status(
            db,
            "gdelt",
            True,
            records_count=inserted_count,
            status_message=f"Synced {len(articles_raw)} articles ({inserted_count} new) at {utc_now().strftime('%H:%M:%S UTC')}",
        )

        return {
            "provider": "gdelt",
            "success": True,
            "fetched": len(articles_raw),
            "inserted": inserted_count,
            "query": query,
        }

    except Exception as e:
        record_provider_status(db, "gdelt", False, str(e))
        return {"provider": "gdelt", "success": False, "error": str(e)[:200], "fetched": 0, "inserted": 0}


# ── Provider 2: Alpha Vantage Ingestion ───────────────────────────────────

def sync_alphavantage_news(
    db: Session,
    topics: str = "economy_macro,energy_transportation",
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Query Alpha Vantage News & Sentiment endpoint.
    Requires ALPHAVANTAGE_API_KEY environment variable.
    """
    api_key = os.getenv("ALPHAVANTAGE_API_KEY", "").strip()
    if not api_key:
        record_provider_status(
            db,
            "alphavantage",
            False,
            error_msg="ALPHAVANTAGE_API_KEY is not configured in server environment (.env).",
        )
        return {"provider": "alphavantage", "configured": False, "success": False, "error": "Not configured"}

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "topics": topics,
        "limit": min(limit, 50),
        "apikey": api_key,
    }

    try:
        code, data = safe_fetch_json(url, params=params, timeout=15)
        if code != 200 or not isinstance(data, dict):
            err = f"Alpha Vantage returned HTTP {code}"
            record_provider_status(db, "alphavantage", False, err)
            return {"provider": "alphavantage", "success": False, "error": err}

        if "feed" not in data:
            note = data.get("Note") or data.get("Information") or data.get("Error Message") or "Rate limited or key invalid"
            record_provider_status(db, "alphavantage", False, note)
            return {"provider": "alphavantage", "success": False, "error": note}

        feed = data["feed"]
        inserted_count = 0

        for item in feed:
            a_title = item.get("title") or "Financial Market Report"
            a_url = item.get("url")
            a_source = item.get("source") or "Alpha Vantage"
            time_pub = item.get("time_published")  # format YYYYMMDDTHHMMSS

            if not a_url or not is_valid_url(a_url):
                continue

            fp = compute_article_fingerprint(a_title, a_url)
            exists = db.query(Article).filter(
                (Article.fingerprint == fp) | (Article.source_url == a_url)
            ).first()
            if exists:
                continue

            pub_dt = utc_now()
            if time_pub:
                try:
                    pub_dt = datetime.strptime(time_pub, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
                except ValueError:
                    pass

            tickers = [t.get("ticker") for t in item.get("ticker_sentiment", []) if t.get("ticker")]
            art_id = f"art_av_{fp[:12]}"
            new_art = Article(
                id=art_id,
                title=a_title[:512],
                source_name=a_source[:256],
                source_url=a_url[:1024],
                published_at=pub_dt,
                retrieved_at=utc_now(),
                excerpt=(item.get("summary") or "")[:1000],
                fingerprint=fp,
                category="Macroeconomics",
                entities=tickers,
                retrieval_status="Source Retrieved",
                review_status="Evidence Reviewed" if tickers else "Pending",
                is_demo=False,
            )
            db.add(new_art)
            inserted_count += 1

        db.commit()
        record_provider_status(
            db,
            "alphavantage",
            True,
            records_count=inserted_count,
            status_message=f"Synced {len(feed)} records ({inserted_count} new) at {utc_now().strftime('%H:%M:%S UTC')}",
        )

        return {"provider": "alphavantage", "success": True, "fetched": len(feed), "inserted": inserted_count}

    except Exception as e:
        record_provider_status(db, "alphavantage", False, str(e))
        return {"provider": "alphavantage", "success": False, "error": str(e)[:200]}


# ── Provider 3: SEC EDGAR Ingestion ────────────────────────────────────────

def sync_sec_edgar(db: Session) -> Dict[str, Any]:
    """
    Test SEC EDGAR submissions endpoint for US large-caps.
    Complies with SEC Fair Access policy with declared User-Agent.
    """
    cik = "0000027904"  # Delta Air Lines (DAL)
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"

    try:
        code, data = safe_fetch_json(url, timeout=12)
        if code == 200 and isinstance(data, dict):
            name = data.get("name", "Delta Air Lines")
            record_provider_status(
                db,
                "sec_edgar",
                True,
                records_count=1,
                status_message=f"Verified SEC EDGAR connection for {name} (CIK {cik}) at {utc_now().strftime('%H:%M:%S UTC')}",
            )
            return {"provider": "sec_edgar", "success": True, "entity": name, "cik": cik}
        else:
            err = f"SEC EDGAR returned HTTP {code}"
            record_provider_status(db, "sec_edgar", False, err)
            return {"provider": "sec_edgar", "success": False, "error": err}
    except Exception as e:
        record_provider_status(db, "sec_edgar", False, str(e))
        return {"provider": "sec_edgar", "success": False, "error": str(e)[:200]}


# ── Master Ingestion Coordinator ──────────────────────────────────────────

def run_news_ingestion(db: Session, query: Optional[str] = None) -> Dict[str, Any]:
    """Run ingestion across all configured real-time providers."""
    q = query or "crude oil OR energy prices OR aviation fuel OR inflation"
    gdelt_res = sync_gdelt_news(db, query=q, max_records=15)
    av_res = sync_alphavantage_news(db, limit=10)
    sec_res = sync_sec_edgar(db)

    total_inserted = (gdelt_res.get("inserted", 0)) + (av_res.get("inserted", 0))
    return {
        "timestamp": utc_now().isoformat(),
        "total_new_articles": total_inserted,
        "providers": {
            "gdelt": gdelt_res,
            "alphavantage": av_res,
            "sec_edgar": sec_res,
        },
    }
