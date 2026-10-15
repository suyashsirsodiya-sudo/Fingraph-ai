"""
data/fetch_edgar.py
===================
Fetch and cache SEC EDGAR Form 10-K filing HTML documents for the companies
listed in data/raw/companies.csv.

This module reads the company list (ticker, cik, url), downloads each 10-K
document if not already cached, and saves it to data/raw/filings/<ticker>.htm.

Usage:
    python -m data.fetch_edgar [--force]

Flags:
    --force    Re-download filings even if already cached locally.

Standing rules:
    - Declares a User-Agent header as required by SEC Fair Access policy.
    - Rate-limits requests to max 5 req/s (configurable via config / .env).
    - Saves raw filings to data/raw/filings/<ticker>.htm for offline use.
    - Never requires external network access when files are already cached.
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

logger = logging.getLogger("fetch_edgar")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_companies(companies_csv: Optional[Path] = None) -> List[Dict]:
    """Load company metadata from data/raw/companies.csv."""
    if companies_csv is None:
        companies_csv = config.COMPANIES_CSV
    if not companies_csv.exists():
        raise FileNotFoundError(f"Companies CSV not found: {companies_csv}")
    companies = []
    with open(companies_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("ticker"):
                companies.append(dict(row))
    return companies


def filing_cache_path(ticker: str, filings_dir: Optional[Path] = None) -> Path:
    """Return the expected local cache path for a ticker's 10-K filing."""
    if filings_dir is None:
        filings_dir = config.FILINGS_DIR
    return filings_dir / f"{ticker}.htm"


def is_cached(ticker: str, filings_dir: Optional[Path] = None) -> bool:
    """Return True if the filing is already cached locally."""
    path = filing_cache_path(ticker, filings_dir)
    return path.exists() and path.stat().st_size > 1000  # must be non-trivial size


def download_filing(
    ticker: str,
    url: str,
    user_agent: str,
    filings_dir: Optional[Path] = None,
    timeout: int = 60,
) -> Path:
    """
    Download a 10-K filing from SEC EDGAR and cache it locally.

    Returns the path to the saved file.
    Raises requests.HTTPError on non-2xx responses.
    """
    if filings_dir is None:
        filings_dir = config.FILINGS_DIR
    filings_dir.mkdir(parents=True, exist_ok=True)

    headers = {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.sec.gov",
    }
    logger.info(f"  Downloading {ticker} 10-K from {url}")
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()

    out_path = filing_cache_path(ticker, filings_dir)
    out_path.write_bytes(resp.content)
    logger.info(f"  Saved {ticker} -> {out_path} ({out_path.stat().st_size:,} bytes)")
    return out_path


def run_fetch(
    force: bool = False,
    companies_csv: Optional[Path] = None,
    filings_dir: Optional[Path] = None,
) -> Dict[str, str]:
    """
    Fetch all 10-K filings for companies in companies.csv.

    Skips already-cached files unless force=True.
    Returns a dict mapping ticker -> local file path (or status string).
    """
    companies = load_companies(companies_csv)
    user_agent = config.SEC_USER_AGENT
    min_delay = 1.0 / config.SEC_MAX_REQ_PER_SEC  # seconds between requests

    results: Dict[str, str] = {}
    download_count = 0

    for company in companies:
        ticker = company.get("ticker", "").strip().upper()
        url = company.get("url", "").strip()
        if not ticker or not url:
            continue

        if not force and is_cached(ticker, filings_dir):
            cached_path = filing_cache_path(ticker, filings_dir)
            logger.info(f"  {ticker}: already cached at {cached_path} ({cached_path.stat().st_size:,} bytes) [SKIP]")
            results[ticker] = str(cached_path)
            continue

        try:
            # Rate limit: wait between requests
            if download_count > 0:
                time.sleep(min_delay)
            path = download_filing(ticker, url, user_agent, filings_dir)
            results[ticker] = str(path)
            download_count += 1
        except Exception as exc:
            logger.error(f"  {ticker}: download failed: {exc}")
            results[ticker] = f"ERROR: {exc}"

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch SEC EDGAR 10-K filings for companies.csv.")
    parser.add_argument("--force", action="store_true", help="Re-download even if already cached")
    args = parser.parse_args()

    print("=== SEC EDGAR 10-K Filing Fetcher ===")
    print(f"User-Agent : {config.SEC_USER_AGENT}")
    print(f"Rate limit : max {config.SEC_MAX_REQ_PER_SEC} req/s")
    print(f"Companies  : {config.COMPANIES_CSV}")
    print(f"Cache dir  : {config.FILINGS_DIR}")
    print(f"Force      : {args.force}")
    print()

    results = run_fetch(force=args.force)
    print("\nResults:")
    for ticker, status in results.items():
        print(f"  {ticker:6s} -> {status}")
    print("\nDone.")
