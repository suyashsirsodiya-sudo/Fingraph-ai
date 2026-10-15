import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from graph.schema import Edge

# Keywords to filter 10-K paragraphs as specified in prompt
KEYWORDS = ["fuel", "oil", "energy", "commodity", "supplier", "customer"]
IMPACT_TERMS = [
    "cost", "price", "expense", "margin", "volatilit", "fluctuat",
    "impact", "affect", "expos", "increas", "decreas", "surcharg",
    "feedstock", "consum", "purchas", "risk", "pressur", "shortag", "inflation"
]

# Rate limiter state
_LAST_REQUEST_TIME = 0.0


def rate_limited_get(url: str, user_agent: str, max_req_per_sec: float) -> requests.Response:
    """Fetch URL with declared User-Agent and max requests per second."""
    global _LAST_REQUEST_TIME
    min_interval = 1.0 / max(0.1, max_req_per_sec)
    now = time.time()
    elapsed = now - _LAST_REQUEST_TIME
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)

    headers = {"User-Agent": user_agent}
    response = requests.get(url, headers=headers, timeout=30)
    _LAST_REQUEST_TIME = time.time()
    return response


def get_cached_filing(ticker: str, filing_date: str, url: str) -> str:
    """Download filing with rate limiting or load from cache."""
    cache_path = config.FILINGS_DIR / f"{ticker}_{filing_date}.htm"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8", errors="ignore")

    print(f"Downloading 10-K for {ticker} from {url}...")
    res = rate_limited_get(url, config.SEC_USER_AGENT, config.SEC_MAX_REQ_PER_SEC)
    if res.status_code != 200:
        raise RuntimeError(f"Failed to fetch {url}: HTTP {res.status_code}")

    html_content = res.text
    cache_path.write_text(html_content, encoding="utf-8", errors="ignore")
    return html_content


def extract_candidate_paragraphs(html_text: str) -> List[str]:
    """
    Extract paragraphs from 10-K HTML mentioning fuel, oil, energy, commodity, supplier or customer.
    Sorts paragraphs by relevance to financial impact and commodity exposure.
    """
    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup(["script", "style", "table", "ix:header"]):
        tag.extract()

    paragraphs: List[str] = []
    seen = set()

    for div in soup.find_all(["div", "p"]):
        # Focus on leaf blocks
        if div.find(["div", "p"]):
            continue
        text = div.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text.split()) < 15 or text in seen:
            continue

        lower_text = text.lower()
        if any(kw in lower_text for kw in KEYWORDS):
            paragraphs.append(text)
            seen.add(text)

    # Score paragraphs so high-relevance risk/cost disclosures are prioritized
    def score_para(p: str) -> int:
        pl = p.lower()
        kw_score = sum(pl.count(kw) for kw in KEYWORDS)
        imp_score = sum(pl.count(term) for term in IMPACT_TERMS)
        return (kw_score * 2) + imp_score

    paragraphs.sort(key=score_para, reverse=True)
    return paragraphs


def call_llm(paragraph: str, company_name: str, ticker: str) -> List[Dict[str, Any]]:
    """
    Call LLM or configurable provider to extract edges.
    Prompt enforces exact verbatim quote extraction from the paragraph.
    Never invents data; returns empty list if no relation stated.
    """
    prompt = (
        f"You are an economic knowledge graph relationship extractor.\n"
        f"Analyze ONLY the following paragraph from an SEC 10-K filing of {company_name} ({ticker}).\n\n"
        f"PARAGRAPH:\n\"{paragraph}\"\n\n"
        f"Extract financial and operational impact edges as a JSON list where events, commodities, "
        f"suppliers, or energy costs affect or expose sectors or companies.\n"
        f"Output schema for each edge:\n"
        f"- src: entity name (e.g. 'Crude Oil', 'Jet Fuel', 'Supplier')\n"
        f"- dst: entity name (e.g. '{ticker}', 'Operating Expenses')\n"
        f"- relation: relationship phrase (e.g. 'is exposed to', 'may affect operating expenses')\n"
        f"- tier: 'fact' | 'inference' | 'hypothesis'\n"
        f"- confidence: number between 0.0 and 1.0\n"
        f"- evidence_quote: EXACT VERBATIM substring copied directly from the paragraph above\n\n"
        f"STRICT RULES:\n"
        f"1. Use ONLY the paragraph provided. Never invent data or quotes.\n"
        f"2. Never use the word 'cause' for inference or hypothesis edges. Say 'may affect' or 'is exposed to'.\n"
        f"3. 'evidence_quote' must be an exact substring of the text.\n"
        f"4. If no explicit relationship is stated, return [].\n"
        f"Respond with JSON ONLY."
    )

    provider = config.LLM_PROVIDER

    # 1. Gemini Provider
    if provider == "gemini" and config.GEMINI_API_KEY:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={config.GEMINI_API_KEY}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"}
            }
            res = requests.post(url, json=payload, timeout=25)
            if res.status_code == 200:
                data = res.json()
                raw_json = data["candidates"][0]["content"]["parts"][0]["text"]
                return json.loads(raw_json)
        except Exception as e:
            print(f"Gemini API error ({e}), falling back to heuristic extractor...")

    # 2. OpenAI Provider
    elif provider == "openai" and config.OPENAI_API_KEY:
        try:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {config.OPENAI_API_KEY}"}
            payload = {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "response_format": {"type": "json_object"}
            }
            res = requests.post(url, headers=headers, json=payload, timeout=25)
            if res.status_code == 200:
                data = res.json()
                content = data["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                return parsed if isinstance(parsed, list) else parsed.get("edges", [])
        except Exception as e:
            print(f"OpenAI API error ({e}), falling back to heuristic extractor...")

    # 3. Exact Verbatim Heuristic Extractor (Deterministic & Verifiable Fallback)
    # Extracts authentic relationship sentences directly from the paragraph without inventing data
    edges = []
    # Split paragraph into sentences preserving exact substrings
    sentence_matches = [m.group(0).strip() for m in re.finditer(r"[^.!?]+[.!?]", paragraph)]
    for sent in sentence_matches:
        sent_lower = sent.lower()
        if not any(k in sent_lower for k in KEYWORDS):
            continue
        if not any(term in sent_lower for term in IMPACT_TERMS):
            continue

        # Determine source entity
        if "jet fuel" in sent_lower or ("fuel" in sent_lower and ticker in ("DAL", "FDX")):
            src = "Jet Fuel"
        elif "diesel" in sent_lower:
            src = "Diesel Fuel"
        elif "crude oil" in sent_lower or "crude" in sent_lower:
            src = "Crude Oil Feedstock"
        elif "feedstock" in sent_lower:
            src = "Petroleum Feedstocks"
        elif "oil" in sent_lower or "petroleum" in sent_lower:
            src = "Crude Oil & Refined Products"
        elif "commodity" in sent_lower:
            src = "Commodity Inputs"
        else:
            src = "Energy & Fuel Inputs"

        dst = ticker
        
        # Determine relation according to standing rules (no 'cause' on inference)
        if "surcharge" in sent_lower:
            relation = "passes through price changes via surcharges to"
        elif "feedstock" in sent_lower or "purchas" in sent_lower:
            relation = "purchases and is exposed to costs of"
        elif "price" in sent_lower or "volatilit" in sent_lower or "fluctuat" in sent_lower:
            relation = "is exposed to price volatility of"
        else:
            relation = "may affect operating expenses of"

        # Tier assignment
        is_fact = bool(re.search(r"(\$|\b\d+\s*\%|\b202\d\b)", sent))
        tier = "fact" if is_fact else "inference"
        confidence = 0.88 if is_fact else 0.76

        # Strict substring check: must be an exact substring of paragraph
        if sent in paragraph:
            edges.append({
                "src": src,
                "dst": dst,
                "relation": relation,
                "tier": tier,
                "confidence": confidence,
                "evidence_quote": sent
            })
            # Limit to at most 2 high-quality edges per paragraph
            if len(edges) >= 2:
                break

    return edges


def verify_quote(quote: str, paragraph: str) -> bool:
    """Step 3: Verify each quote is an exact substring of the paragraph."""
    if not quote or not quote.strip():
        return False
    return quote.strip() in paragraph


def run_pipeline(max_paras_per_company: int = 15):
    """Main execution pipeline."""
    if not config.COMPANIES_CSV.exists():
        raise FileNotFoundError(f"Missing {config.COMPANIES_CSV}")

    with open(config.COMPANIES_CSV, mode="r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    print(f"Loaded {len(reader)} companies from {config.COMPANIES_CSV.name}")
    print(f"Using SEC User-Agent: '{config.SEC_USER_AGENT}' (Max {config.SEC_MAX_REQ_PER_SEC} req/s)")
    print(f"LLM Provider: '{config.LLM_PROVIDER}'")

    total_paras_processed = 0
    total_edges_kept = 0
    total_edges_dropped = 0
    valid_edges: List[Edge] = []
    companies_covered = set()

    # Clear or open pipeline log
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    pipeline_log_file = open(config.PIPELINE_LOG, mode="w", encoding="utf-8")

    for row in reader:
        ticker = row["ticker"].strip()
        name = row["company_name"].strip()
        filing_date = row.get("filing_date", "unknown").strip()
        url = row["url"].strip()

        print(f"\nProcessing {name} ({ticker})...")
        try:
            html = get_cached_filing(ticker, filing_date, url)
        except Exception as e:
            print(f"Error fetching 10-K for {ticker}: {e}")
            continue

        paragraphs = extract_candidate_paragraphs(html)
        print(f"Found {len(paragraphs)} keyword-matching candidate paragraphs.")

        selected_paragraphs = paragraphs[:max_paras_per_company]

        for p_idx, para in enumerate(selected_paragraphs):
            total_paras_processed += 1
            extracted_edges = call_llm(para, name, ticker)

            if not extracted_edges:
                # Log paragraph with no edge
                log_entry = {
                    "source": url,
                    "company": ticker,
                    "relation_type": "none",
                    "quote_status": "needs_source",
                    "tier": "none",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                pipeline_log_file.write(json.dumps(log_entry) + "\n")
                continue

            for raw_edge in extracted_edges:
                quote = raw_edge.get("evidence_quote", "")
                relation = raw_edge.get("relation", "is exposed to")
                tier = raw_edge.get("tier", "inference")
                confidence = float(raw_edge.get("confidence", 0.7))

                # Step 3: Exact substring check
                is_verified = verify_quote(quote, para)

                if is_verified:
                    quote_status = "verified"
                    # Standing rule 3: never use cause for inference/hypothesis
                    if tier in ("inference", "hypothesis") and "cause" in relation.lower():
                        relation = re.sub(r"\bcause[sd]?\b", "may affect", relation, flags=re.IGNORECASE)

                    confidence = max(0.0, min(1.0, confidence))

                    try:
                        edge_obj = Edge(
                            src=raw_edge.get("src", "Unknown Source"),
                            dst=raw_edge.get("dst", ticker),
                            relation=relation,
                            tier=tier if tier in ("fact", "inference", "hypothesis") else "inference",
                            confidence=confidence,
                            evidence_quote=quote.strip(),
                            source_url=url,
                            source_date=filing_date,
                            company=ticker
                        )
                        valid_edges.append(edge_obj)
                        companies_covered.add(ticker)
                        total_edges_kept += 1
                    except Exception as err:
                        quote_status = "rejected"
                        total_edges_dropped += 1
                else:
                    quote_status = "rejected"
                    total_edges_dropped += 1

                # Step 4: Append one line per paragraph/edge to data/pipeline_log.jsonl
                log_entry = {
                    "source": url,
                    "company": ticker,
                    "relation_type": relation,
                    "quote_status": quote_status,
                    "tier": tier,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                pipeline_log_file.write(json.dumps(log_entry) + "\n")

    pipeline_log_file.close()

    # Step 5: Save valid edges to data/edges.json using graph/schema.Edge
    edges_data = [edge.model_dump() for edge in valid_edges]
    with open(config.EDGES_JSON, mode="w", encoding="utf-8") as f:
        json.dump(edges_data, f, indent=2)

    # Step 6: Print required summary
    print("\n" + "=" * 50)
    print("EXTRACTION SUMMARY")
    print("=" * 50)
    print(f"paragraphs processed: {total_paras_processed}")
    print(f"edges kept: {total_edges_kept}")
    print(f"edges dropped: {total_edges_dropped}")
    print(f"companies covered: {len(companies_covered)} ({', '.join(sorted(companies_covered))})")
    print(f"saved edges to: {config.EDGES_JSON}")
    print(f"pipeline log saved to: {config.PIPELINE_LOG}")
    print("=" * 50)

    # Done criteria verification
    assert len(companies_covered) >= 5, f"Expected at least 5 companies covered, got {len(companies_covered)}"
    assert config.PIPELINE_LOG.exists(), "pipeline_log.jsonl does not exist"
    for e in valid_edges:
        assert e.evidence_quote, "Kept edge has empty quote"
        assert e.source_url, "Kept edge has empty source_url"
        assert e.tier in ("fact", "inference", "hypothesis"), f"Invalid tier {e.tier}"
        if e.tier in ("inference", "hypothesis"):
            assert "cause" not in e.relation.lower(), f"Forbidden word 'cause' in relation {e.relation}"
    print("ALL VALIDATION CRITERIA SATISFIED!")


if __name__ == "__main__":
    run_pipeline()
