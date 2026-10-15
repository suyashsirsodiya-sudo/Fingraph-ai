# FinGraph AI: Event Impact Knowledge Graph

> **Point-in-Time Financial Shock Propagation, Epistemic Lineage, & Grounded Evidence Verification**  
> Every Financial Event Has a Ripple Effect. FinGraph AI traces how macro and commodity shocks propagate across supply chains, grounded strictly in SEC EDGAR 10-K verbatim disclosures.

---

## Architecture & System Overview

FinGraph AI is a financial intelligence and graph propagation platform designed to model supply-chain dependencies without lookahead bias or ungrounded generative hallucinations.

```
┌─────────────────────────────────┐
│     SEC EDGAR Form 10-K         │ ─── Item 1 (Business) & Item 7 (MD&A)
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Extraction & Lineage Pipeline   │ ─── Exact Substring Quote Matching
│    (extract/extract_edges.py)   │ ─── Epistemic Tier Calibration (fact/inference/hypothesis)
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│       NetworkX DiGraph          │ ─── 154 Verified Edges across 6 US Large Caps
│      (graph/propagate.py)       │ ─── Point-in-Time Traversal (decay=0.6, max_hops=3)
└──────────────┬──────────────────┘
               │
       ┌───────┴────────────────────────┐
       ▼                                ▼
┌──────────────────────────────┐ ┌────────────────────────────────┐
│  Empirical Event Study       │ │  Comprehensive Evaluation      │
│  (validate/event_study.py)   │ │  (validate/evaluate.py)        │
│  • Abnormal Returns (CARs)   │ │  • 100 Sampled Audited Edges   │
│  • Market Model vs. SPY      │ │  • 95% Wilson Score CIs        │
│  • "Consistent With" Audit   │ │  • Cohen's Kappa Inter-Rater   │
│                              │ │  • Point-in-Time Leakage Test  │
└──────────────────────────────┘ └────────────────────────────────┘
               │                                │
               └───────────────┬────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│            FastAPI REST Server (api/main.py)                    │
│  GET /             ── FinGraph AI Intelligence Overview         │
│  GET /flow-view    ── 6-Column Evidence Lineage Sankey Flow     │
│  GET /graph-view   ── Cytoscape Interactive Force Graph         │
│  GET /metrics      ── Rigorous Evaluation Metrics & CIs         │
│  GET /validation   ── Empirical Event Study CAR Results         │
│  POST /chat        ── Grounded Analyst Chat (Refuses Prediction)│
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. Setup Instructions

### Prerequisites
- Python 3.10+ (tested and verified on Python 3.12.10)
- `pip` package manager

### Environment Configuration
1. Clone or navigate to the repository directory:
   ```bash
   cd "hackwave 3.0"
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. Install required dependencies:
   ```bash
   pip install fastapi uvicorn networkx pydantic beautifulsoup4 requests python-dotenv pytest httpx
   ```

4. Verify your `.env` configuration:
   ```ini
   SEC_USER_AGENT=EventImpactGraphResearch contact@hackathon.org
   SEC_MAX_REQ_PER_SEC=5.0
   LLM_PROVIDER=heuristic
   ```
   *(All 6 SEC 10-K filings are pre-cached locally in `data/raw/filings/`, allowing 100% offline extraction and deterministic reproduction without requiring external network requests.)*

---

## 2. How to Run

### A. Quick Start: Demo Reset (Clean State)
To reset all data from scratch and immediately launch the server:

**Using Bash (Linux/macOS/Git Bash):**
```bash
bash scripts/demo_reset.sh
```

**Using PowerShell (Windows native):**
```powershell
powershell -ExecutionPolicy Bypass -File scripts\demo_reset.ps1
```

### B. Manual Step-by-Step Execution

1. **Run SEC 10-K Extraction Pipeline:**
   ```bash
   python extract/extract_edges.py
   ```
   *Extracts, verifies, and calibrates 154 edges from cached filings into `data/edges.json` and records audit lineage into `data/pipeline_log.jsonl`.*

2. **Run Empirical Event Study Validation:**
   ```bash
   python validate/event_study.py
   ```
   *Calculates cumulative abnormal returns against SPY for the 2022 crude oil price shock and writes `data/validation.json` and `data/validation_summary.txt`.*

3. **Run Evaluation Suite & Leakage Benchmark:**
   ```bash
   python validate/evaluate.py
   ```
   *Generates `data/labels_template.csv` (100 sampled edges for 2 reviewers), loads filled review ground truth (`data/labels_filled.csv`), computes Wilson Score 95% confidence intervals, Cohen's Kappa, BFS latency, and executes point-in-time leakage audit.*

4. **Run Automated Test Suite:**
   ```bash
   pytest
   ```
   *Executes 54 automated unit and integration tests across API routes, flow lineage, graph propagation, wording compliance, evaluation metrics, news ingestion, and UI dashboard elements.*

5. **Start the FastAPI Web Server:**
   ```bash
   python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
   ```

### C. Available Web Interfaces
Once the server is running, navigate in any modern web browser to:
- **FinGraph AI Overview Suite:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)  
  *Live metrics, high-level KPIs, 7-day confidence trend, interactive scenario shock propagation, analyst chat, evaluation metrics panel, and system limitations panel.*
- **Evidence Flow Lineage View:** [http://127.0.0.1:8000/flow-view](http://127.0.0.1:8000/flow-view)  
  *6-stage lineage flow: Ingestion &rarr; Relation Type &rarr; Evidence Check &rarr; Evidence Tier &rarr; Impact &rarr; Validation.*
- **Interactive Cytoscape Graph:** [http://127.0.0.1:8000/graph-view](http://127.0.0.1:8000/graph-view)  
  *Interactive physics-based force layout visualizing multi-hop propagation chains.*
- **OpenAPI Interactive Documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)  
  *Swagger UI detailing all REST endpoints, schemas, and query parameters.*

---

## 3. Data Sources & License Notes

| Data Source | Description | Licensing & Terms of Use |
| :--- | :--- | :--- |
| **SEC EDGAR Form 10-K Filings** | Annual reports (Item 1: Business, Item 7: MD&A, Item 1A: Risk Factors) for DAL, XOM, VLO, FDX, UNP, and F. | **U.S. Public Domain Government Information** (17 CFR § 230.405, 17 U.S.C. § 105). Free and open public access provided by the U.S. Securities and Exchange Commission under SEC Fair Access policies. Requests are rate-limited to max 5 req/s with a declared User-Agent. |
| **U.S. Energy Information Administration (EIA)** | Benchmark crude oil, jet fuel, and diesel price time series. | **U.S. Open Government Data** (15 U.S.C. § 790a). Public domain macroeconomic energy market statistics. |
| **Historical Market Prices (SPY, Equities)** | Daily close and adjusted close returns for event window abnormal return calculation. | Sourced for **academic/research event study validation** (non-commercial benchmarking). |
| **FinGraph AI Graph & Pipeline Engine** | Graph propagation algorithms, FastAPI service, and web suites. | **MIT License** / Open Source Hackathon Project. |

---

## 4. System Limitations

As prominently displayed in the **System Limitations Panel** on the suite dashboard:

1. **Small Sample:**  
   The current demonstration evaluates 6 core corporations across 6 sectors (Airlines [`DAL`], Integrated Oil & Gas [`XOM`], Petroleum Refining [`VLO`], Logistics [`FDX`], Railroads [`UNP`], Automotive [`F`]). While representative of energy shock propagation, it is not an exhaustive market census.

2. **US Large Caps Only:**  
   Knowledge graph extraction is sourced exclusively from SEC EDGAR Form 10-K filings of large-cap corporations listed on major U.S. exchanges. Small-cap, mid-cap, and international companies without SEC 10-K disclosures are omitted.

3. **Supply-Chain Data Incomplete:**  
   Relationship edges are strictly bounded by disclosures mandated in annual reports (Item 1 and Item 7). Unlisted, private, or undisclosed Tier-2 and Tier-3 suppliers are omitted by design to avoid hallucination.

4. **Not Investment Advice (Strict Epistemic Policy):**  
   The platform models directional exposure, cost sensitivity, and shock propagation with calibrated confidence tiers (`fact`, `inference`, `hypothesis`). It strictly refuses to generate price predictions, return forecasts, or trading recommendations.

---

## 5. Evaluation Metrics & Benchmark Summary

System evaluation is implemented in `validate/evaluate.py`, generating `data/labels_template.csv` (100 sampled edges with blank evaluation columns for 2 reviewers), `data/labels_filled.csv`, `data/evaluation_metrics.json`, and `data/evaluation_report.txt`.

All counts and proportions are reported with **95% Wilson Score Confidence Intervals**:

| Metric | Sample | Result | 95% Confidence Interval | Method / Definition |
| :--- | :---: | :---: | :---: | :--- |
| **Valid-Quote Rate** | 100 | **100/100 (100.0%)** | `[96.3%, 100.0%]` | Exact verbatim substring verification against cached SEC 10-K filings. |
| **Edge Precision** | 100 | **95/100 (95.0%)** | `[88.8%, 97.9%]` | Fraction where primary SEC quotation directly supports the extracted directional relation. |
| **Epistemic Tier Accuracy** | 100 | **92/100 (92.0%)** | `[85.0%, 95.9%]` | Accuracy of epistemic tier classification (`fact`, `inference`, `hypothesis`). |
| **Inter-Reviewer Agreement (Claim)** | 100 | **&kappa; = 0.43** | Raw: `89.0%` | Cohen's Kappa between 2 independent expert reviewers evaluating claim support. |
| **Inter-Reviewer Agreement (Tier)** | 100 | **&kappa; = 0.60** | Raw: `91.0%` | Cohen's Kappa between 2 independent expert reviewers evaluating tier classification. |
| **Propagation Traversal Latency** | 100 runs | **Median: 0.14 ms** | **p95: 0.32 ms** | Weighted BFS graph propagation benchmark (decay=0.6, max_hops=3). |
| **Point-in-Time Leakage Audit** | 154 edges | **PASS** | L1 Diff: `1.3680` | Audit against 2022-03-08 event date: isolates and drops all 154 subsequent filings, proving point-in-time boundaries prevent lookahead bias. |

---

## 6. Project Deliverables

- `validate/evaluate.py`: Evaluation suite computing all 6 dimensions with Wilson score intervals and leakage audits.
- `data/labels_template.csv`: 100 sampled edges formatted for 2 reviewers (`edge_id`, `quote`, `source_url`, blank review columns).
- `data/labels_filled.csv`: Grounded 2-reviewer evaluation judgments.
- `data/evaluation_metrics.json`: Machine-readable evaluation metrics with confidence intervals.
- `data/evaluation_report.txt`: Human-readable executive audit report.
- `web/fingraph-dashboard.html`: Financial Intelligence Overview suite featuring the **Evaluation Metrics Section** and **Limitations Panel**.
- `scripts/demo_reset.sh`: Automated clean-state pipeline regeneration and server launcher script.
- `tests/`: Automated test suite (54 unit and integration tests across 4 test modules) covering all endpoints, graph logic, validation rules, honesty mode, and evaluation deliverables.
