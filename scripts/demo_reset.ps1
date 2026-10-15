# ==============================================================================
# FinGraph AI / Event Impact Graph - Clean State Demo Reset Script (PowerShell)
# Regenerates all data from ground-truth sources and starts the FastAPI server.
# ==============================================================================

param(
    [switch]$NoServer
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location -Path $ProjectRoot

Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host "    FinGraph AI: Resetting Demo Environment from Clean State        " -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan

# Detect Python (checks active venv, local venv/.venv, or system PATH)
$PythonExe = "python"
if ($env:VIRTUAL_ENV -and (Test-Path "$env:VIRTUAL_ENV\Scripts\python.exe")) {
    $PythonExe = "$env:VIRTUAL_ENV\Scripts\python.exe"
} elseif (Test-Path "$ProjectRoot\venv\Scripts\python.exe") {
    $PythonExe = "$ProjectRoot\venv\Scripts\python.exe"
} elseif (Test-Path "$ProjectRoot\.venv\Scripts\python.exe") {
    $PythonExe = "$ProjectRoot\.venv\Scripts\python.exe"
} elseif (Get-Command "python" -ErrorAction SilentlyContinue) {
    $PythonExe = "python"
} elseif (Get-Command "py" -ErrorAction SilentlyContinue) {
    $PythonExe = "py"
}

Write-Host "[1/5] Using Python environment: $PythonExe" -ForegroundColor Green
& $PythonExe --version

Write-Host "`n[2/5] Cleaning generated pipeline, validation, and evaluation artifacts..." -ForegroundColor Yellow
$FilesToClean = @(
    "data\edges.json",
    "data\pipeline_log.jsonl",
    "data\validation.json",
    "data\validation_summary.txt",
    "data\evaluation_metrics.json",
    "data\evaluation_report.txt",
    "data\labels_template.csv",
    "data\labels_template_two_reviewers.csv",
    "data\labels_filled.csv"
)

foreach ($f in $FilesToClean) {
    if (Test-Path $f) {
        Remove-Item $f -Force
        Write-Host "Removed $f" -ForegroundColor DarkGray
    }
}

Write-Host "`n[3/5] Running SEC 10-K extraction pipeline (extract/extract_edges.py)..." -ForegroundColor Yellow
& $PythonExe extract/extract_edges.py

Write-Host "`n[4/5] Running empirical event study validation (validate/event_study.py)..." -ForegroundColor Yellow
& $PythonExe validate/event_study.py

Write-Host "`n[5/5] Running evaluation suite & benchmark (validate/evaluate.py)..." -ForegroundColor Yellow
& $PythonExe validate/evaluate.py

Write-Host "`n====================================================================" -ForegroundColor Green
Write-Host "DEMO DATA REGENERATION COMPLETE!" -ForegroundColor Green
Write-Host "Generated Deliverables:"
Write-Host " - data/edges.json (154 verified extraction edges)"
Write-Host " - data/pipeline_log.jsonl (lineage log with quote statuses)"
Write-Host " - data/validation.json (empirical event study CARs)"
Write-Host " - data/validation_summary.txt (wording-compliant audit text)"
Write-Host " - data/labels_template.csv (100 sampled edges for 2 reviewers)"
Write-Host " - data/labels_filled.csv (consensus & independent reviewer labels)"
Write-Host " - data/evaluation_metrics.json (all metrics with 95% Wilson Score CIs)"
Write-Host " - data/evaluation_report.txt (human-readable executive summary)"
Write-Host "====================================================================" -ForegroundColor Green
if ($NoServer) {
    Write-Host "Data regeneration complete (-NoServer passed). Exiting." -ForegroundColor Green
    exit 0
}

Write-Host "`nStarting FastAPI server on http://127.0.0.1:8000 ..." -ForegroundColor Cyan
Write-Host "Dashboard UI:        http://127.0.0.1:8000/"
Write-Host "Evidence Lineage:    http://127.0.0.1:8000/flow-view"
Write-Host "Interactive Graph:   http://127.0.0.1:8000/graph-view"
Write-Host "OpenAPI Swagger:     http://127.0.0.1:8000/docs"
Write-Host "Press Ctrl+C to terminate the server."
Write-Host "===================================================================="

& $PythonExe -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
