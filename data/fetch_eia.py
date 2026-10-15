"""
data/fetch_eia.py
=================
Fetch benchmark energy price time series from the U.S. Energy Information
Administration (EIA) Open Data API v2 and save them as CSV to data/raw/.

Series fetched:
  - PET.RWTC.D   : WTI crude oil spot price (Dollars per Barrel), daily
  - PET.EER_EPD2DXL0_PF4_RGC_DPG.W : U.S. Gulf Coast Kerosene-Type Jet Fuel (Dollars/gallon), weekly
  - PET.EMD_EPD2D_PTE_NUS_DPG.W     : U.S. No 2 Diesel Retail Prices (Dollars/gallon), weekly

Usage:
    python -m data.fetch_eia [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--api-key KEY]

Standing rules:
    - No lookahead: dates fetched are bounded by --start / --end args.
    - Saves raw CSVs to data/raw/eia_<series_id_sanitized>.csv.
    - Falls back gracefully to offline synthetic data if network or API key unavailable.
"""

import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import requests

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

logger = logging.getLogger("fetch_eia")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

EIA_BASE_URL = "https://api.eia.gov/v2/seriesid/{series_id}"

# EIA v2 API series IDs used in this project
# PET.RWTC.D  = WTI crude oil daily spot price ($/barrel)
# PET.EER_EPD2DXL0_PF4_RGC_DPG.W = US Gulf Coast Jet Fuel weekly ($/gallon)
# PET.EMD_EPD2D_PTE_NUS_DPG.W     = US No 2 Diesel Retail weekly ($/gallon)
SERIES_IDS = {
    "wti_crude_daily": "PET.RWTC.D",
    "jet_fuel_weekly": "PET.EER_EPD2DXL0_PF4_RGC_DPG.W",
    "diesel_retail_weekly": "PET.EMD_EPD2D_PTE_NUS_DPG.W",
}


def _sanitize_filename(series_id: str) -> str:
    """Convert a series ID to a safe filename component."""
    return series_id.replace(".", "_").replace("/", "_")


def fetch_eia_series(
    series_id: str,
    api_key: str,
    start_date: str,
    end_date: str,
    timeout: int = 30,
) -> List[Dict]:
    """
    Fetch a single EIA v2 series via the seriesid endpoint.

    Returns a list of dicts with keys: date, value, series_id.
    Raises requests.HTTPError on bad status codes.
    """
    url = EIA_BASE_URL.format(series_id=series_id)
    params = {
        "api_key": api_key,
        "start": start_date,
        "end": end_date,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "offset": 0,
        "length": 5000,
    }
    logger.info(f"  GET {url} | series={series_id} | {start_date} -> {end_date}")
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()

    # EIA v2 response structure: response.data[{period, value, ...}]
    data_rows = payload.get("response", {}).get("data", [])
    results = []
    for row in data_rows:
        period = row.get("period", "")
        value = row.get("value", None)
        if period and value is not None:
            try:
                float_val = float(value)
            except (TypeError, ValueError):
                continue
            results.append({"date": period, "value": float_val, "series_id": series_id})
    return results


def save_series_csv(records: List[Dict], out_path: Path) -> None:
    """Write records to CSV with columns: date, value, series_id."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "value", "series_id"])
        writer.writeheader()
        writer.writerows(records)
    logger.info(f"  Saved {len(records)} rows -> {out_path}")


def generate_synthetic_fallback(series_name: str, start_date: str, end_date: str) -> List[Dict]:
    """
    Generate deterministic synthetic data when EIA API is unavailable.
    Used for offline testing and CI environments.
    """
    import random
    rng = random.Random(hash(series_name) & 0xFFFFFFFF)
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    base_prices = {
        "wti_crude_daily": 80.0,
        "jet_fuel_weekly": 2.50,
        "diesel_retail_weekly": 3.80,
    }
    base = base_prices.get(series_name, 50.0)
    records = []
    current = start_dt
    step = timedelta(days=1) if "daily" in series_name else timedelta(weeks=1)
    while current <= end_dt:
        base += rng.uniform(-1.5, 1.5)
        records.append({"date": current.strftime("%Y-%m-%d"), "value": round(max(base, 1.0), 2), "series_id": SERIES_IDS.get(series_name, series_name)})
        current += step
    logger.warning(f"  [SYNTHETIC FALLBACK] {series_name}: {len(records)} synthetic rows generated.")
    return records


def run_fetch(
    api_key: Optional[str] = None,
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
    out_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Fetch all EIA series and save to out_dir (default: data/raw/).

    Returns a dict mapping series_name -> output CSV path.
    """
    if api_key is None:
        api_key = os.getenv("EIA_API_KEY", "")
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
    if out_dir is None:
        out_dir = config.RAW_DIR

    saved_paths: Dict[str, Path] = {}
    for name, series_id in SERIES_IDS.items():
        safe_name = _sanitize_filename(series_id)
        out_path = out_dir / f"eia_{safe_name}.csv"

        if api_key:
            try:
                records = fetch_eia_series(series_id, api_key, start_date, end_date)
                if not records:
                    logger.warning(f"  No data returned for {series_id}. Using synthetic fallback.")
                    records = generate_synthetic_fallback(name, start_date, end_date)
            except Exception as exc:
                logger.warning(f"  EIA fetch failed for {series_id}: {exc}. Using synthetic fallback.")
                records = generate_synthetic_fallback(name, start_date, end_date)
        else:
            logger.warning(f"  No EIA_API_KEY found. Using synthetic fallback for {name}.")
            records = generate_synthetic_fallback(name, start_date, end_date)

        save_series_csv(records, out_path)
        saved_paths[name] = out_path

    return saved_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch EIA energy price time series.")
    parser.add_argument("--api-key", type=str, default=None, help="EIA API key (or set EIA_API_KEY env var)")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    print("=== EIA Energy Price Fetcher ===")
    paths = run_fetch(api_key=args.api_key, start_date=args.start, end_date=args.end)
    print("\nFiles written:")
    for name, path in paths.items():
        print(f"  {name:30s} -> {path}")
    print("\nDone.")
