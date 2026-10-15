#!/usr/bin/env bash
# ==============================================================================
# FinGraph AI / Event Impact Graph - Clean State Demo Reset Script
# Regenerates all data from ground-truth sources and starts the FastAPI server.
# ==============================================================================

set -e

# Resolve project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "===================================================================="
echo "    FinGraph AI: Resetting Demo Environment from Clean State        "
echo "===================================================================="

# Detect working Python 3 executable (virtualenv, python3, python, py)
PYTHON_CMD=""
for candidate in \
    "$VIRTUAL_ENV/bin/python" \
    "$VIRTUAL_ENV/Scripts/python.exe" \
    "$PROJECT_ROOT/venv/bin/python" \
    "$PROJECT_ROOT/venv/Scripts/python.exe" \
    "$PROJECT_ROOT/.venv/bin/python" \
    "$PROJECT_ROOT/.venv/Scripts/python.exe" \
    "python3" \
    "python" \
    "py"; do
    if [ -n "$candidate" ] && "$candidate" --version &>/dev/null; then
        PYTHON_CMD="$candidate"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "ERROR: Working Python 3 executable not found." >&2
    exit 1
fi

echo "[1/5] Using Python environment:"
"$PYTHON_CMD" --version

# Step 1: Clean generated artifacts (preserving cached SEC filings in data/raw)
echo ""
echo "[2/5] Cleaning generated pipeline, validation, and evaluation artifacts..."
rm -f data/edges.json
rm -f data/pipeline_log.jsonl
rm -f data/validation.json
rm -f data/validation_summary.txt
rm -f data/evaluation_metrics.json
rm -f data/evaluation_report.txt
rm -f data/labels_template.csv
rm -f data/labels_template_two_reviewers.csv
rm -f data/labels_filled.csv

# Step 2: Regenerate edges and pipeline log from SEC 10-K filings
echo ""
echo "[3/5] Running SEC 10-K extraction pipeline (extract/extract_edges.py)..."
"$PYTHON_CMD" extract/extract_edges.py

# Step 3: Run empirical event study validation
echo ""
echo "[4/5] Running empirical event study validation (validate/event_study.py)..."
"$PYTHON_CMD" validate/event_study.py

# Step 4: Run comprehensive evaluation suite (metrics, reviewers, leakage test)
echo ""
echo "[5/5] Running evaluation suite & benchmark (validate/evaluate.py)..."
"$PYTHON_CMD" validate/evaluate.py

echo ""
echo "===================================================================="
echo "DEMO DATA REGENERATION COMPLETE!"
echo "Generated Deliverables:"
echo " - data/edges.json (154 verified extraction edges)"
echo " - data/pipeline_log.jsonl (lineage log with quote statuses)"
echo " - data/validation.json (empirical event study CARs)"
echo " - data/validation_summary.txt (wording-compliant audit text)"
echo " - data/labels_template.csv (100 sampled edges for 2 reviewers)"
echo " - data/labels_filled.csv (consensus & independent reviewer labels)"
echo " - data/evaluation_metrics.json (all metrics with 95% Wilson Score CIs)"
echo " - data/evaluation_report.txt (human-readable executive summary)"
echo "===================================================================="
echo ""
if [ "$1" = "--no-server" ]; then
    echo "Data regeneration complete (--no-server passed). Exiting."
    exit 0
fi

echo "Starting FastAPI server on http://127.0.0.1:8000 ..."
echo "Dashboard UI:        http://127.0.0.1:8000/"
echo "Evidence Lineage:    http://127.0.0.1:8000/flow-view"
echo "Interactive Graph:   http://127.0.0.1:8000/graph-view"
echo "OpenAPI Swagger:     http://127.0.0.1:8000/docs"
echo "Press Ctrl+C to terminate the server."
echo "===================================================================="

exec "$PYTHON_CMD" -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
