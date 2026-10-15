import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Project directory structure
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
FILINGS_DIR = RAW_DIR / "filings"
COMPANIES_CSV = RAW_DIR / "companies.csv"
EDGES_JSON = DATA_DIR / "edges.json"
PIPELINE_LOG = DATA_DIR / "pipeline_log.jsonl"
EVENTS_CSV = DATA_DIR / "events.csv"
VALIDATION_JSON = DATA_DIR / "validation.json"
SHOCK_CONFIG = BASE_DIR / "graph" / "shock_config.json"

# Ensure directories exist
RAW_DIR.mkdir(parents=True, exist_ok=True)
FILINGS_DIR.mkdir(parents=True, exist_ok=True)

# SEC EDGAR Settings (Standing rule: declared User-Agent from .env, max 5 req/s)
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "EventImpactGraphResearch contact@hackathon.org")
SEC_MAX_REQ_PER_SEC = float(os.getenv("SEC_MAX_REQ_PER_SEC", "5.0"))

# LLM Settings (configurable provider)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "heuristic").lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
