# Campus Sentinel: check everything, install what is missing, run backend + dashboard
# in ONE terminal. Ctrl+C stops both.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 -CheckOnly
#   powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 -Model llama3.1:8b
#   powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 -SkipOllama

param(
    [switch]$CheckOnly,
    [switch]$SkipOllama,
    [string]$Model = "llama3.2:3b"
)

# "Continue", not "Stop": Windows PowerShell 5.1 turns any stderr text from python/pip/npm
# into a fatal error under "Stop". Failures are detected through $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
$Missing = @()

function Ok($msg)   { Write-Host "  [OK]      $msg" -ForegroundColor Green }
function Miss($msg) { Write-Host "  [MISSING] $msg" -ForegroundColor Yellow }
function Bad($msg)  { Write-Host "  [ERROR]   $msg" -ForegroundColor Red }
function Step($msg) { Write-Host "`n== $msg" -ForegroundColor Cyan }

# ---------------------------------------------------------------- 1. tools
Step "Checking tools"
$PyCmd = $null
foreach ($cand in @("python", "py -3.11", "py -3.12", "py -3.13", "py -3.10", "py")) {
    $parts = $cand.Split(" ")
    try {
        $v = & $parts[0] $parts[1..($parts.Length)] --version 2>&1 | Out-String
        if ($LASTEXITCODE -eq 0 -and $v -match "Python 3\.(\d+)") {
            if ([int]$Matches[1] -ge 10 -and [int]$Matches[1] -le 13) { $PyCmd = $parts; Ok "$($v.Trim()) ($cand)"; break }
        }
    } catch { }
}
if (-not $PyCmd) {
    if (Test-Path $VenvPy) { Ok "Python found in existing .venv" }
    else { Bad "Python 3.10-3.13 not found (python --version). Install Python 3.11+ from python.org and tick 'Add to PATH'"; exit 1 }
}
if (Get-Command node -ErrorAction SilentlyContinue) { Ok "Node $(node -v)" }
else { Bad "Node.js not found. Install Node 20+ from nodejs.org, then restart VS Code"; exit 1 }
$HasOllama = [bool](Get-Command ollama -ErrorAction SilentlyContinue)
if ($SkipOllama) { Write-Host "  [SKIP]    Ollama (reports + chat disabled)" }
elseif ($HasOllama) { Ok "Ollama installed" }
else { Miss "Ollama not installed: reports + chat won't work (https://ollama.com/download)" }

# ---------------------------------------------------------------- 2. venv + packages
Step "Checking Python packages"
if (Test-Path $VenvPy) { Ok ".venv exists" }
else {
    Miss ".venv"; $Missing += "venv"
    if (-not $CheckOnly) {
        & $PyCmd[0] $PyCmd[1..($PyCmd.Length)] -m venv .venv
        Ok ".venv created"
    }
}
$NeedPip = $true
if (Test-Path $VenvPy) {
    & $VenvPy -c "import fastapi, uvicorn, sqlalchemy, multipart, httpx, cv2, numpy, ultralytics, chromadb, ollama, flwr, matplotlib" 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) { Ok "all Python packages installed"; $NeedPip = $false }
}
if ($NeedPip) {
    Miss "Python packages"; $Missing += "pip"
    if (-not $CheckOnly) {
        Write-Host "  installing (first time ~10 min, PyTorch is large)..."
        & $VenvPy -m pip install --upgrade pip
        & $VenvPy -m pip install -r backend\requirements.txt
        if ($LASTEXITCODE -ne 0) { Bad "pip install failed (see above)"; exit 1 }
        Ok "Python packages installed"
    }
}

# ---------------------------------------------------------------- 3. frontend packages
Step "Checking dashboard packages"
if ((Test-Path "frontend\node_modules\vite") -and (Test-Path "frontend\node_modules\recharts") -and (Test-Path "frontend\node_modules\react-router-dom")) { Ok "node_modules present" }
else {
    Miss "frontend\node_modules"; $Missing += "npm"
    if (-not $CheckOnly) {
        Push-Location frontend; npm install; $code = $LASTEXITCODE; Pop-Location
        if ($code -ne 0) { Bad "npm install failed"; exit 1 }
        Ok "npm packages installed"
    }
}

# ---------------------------------------------------------------- 4. federated model
Step "Checking federated model"
if (Test-Path "federated\artifacts\global_model.json") { Ok "global model present" }
else {
    Miss "federated\artifacts\global_model.json"; $Missing += "fl"
    if (-not $CheckOnly) {
        & $VenvPy -m federated.data --clients 5 | Out-Null
        & $VenvPy -m federated.simulate --quiet
        Ok "global model trained"
    }
}

# ---------------------------------------------------------------- 5. ollama + RAG index
if (-not $SkipOllama -and $HasOllama) {
    Step "Checking Ollama + RAG knowledge base"
    $tags = $null
    try { $tags = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 3 } catch { }
    if (-not $tags) {
        Miss "Ollama server not running"
        if (-not $CheckOnly) {
            Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
            for ($i = 0; $i -lt 15 -and -not $tags; $i++) {
                Start-Sleep 1
                try { $tags = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 2 } catch { }
            }
            if ($tags) { Ok "Ollama started" } else { Bad "could not start Ollama. Open the Ollama app, then rerun" }
        }
    } else { Ok "Ollama server running" }

    if ($tags) {
        $names = @($tags.models | ForEach-Object { $_.name })
        foreach ($m in @("nomic-embed-text", $Model)) {
            if ($names | Where-Object { $_ -like "$m*" }) { Ok "model $m" }
            else {
                Miss "model $m"; $Missing += $m
                if (-not $CheckOnly) { ollama pull $m }
            }
        }
        $cfg = Get-Content rag\config.py -Raw
        if ($cfg -notmatch [regex]::Escape("LLM_MODEL = `"$Model`"")) {
            if ($CheckOnly) { Miss "rag\config.py LLM_MODEL is not $Model" }
            else {
                $cfg = $cfg -replace 'LLM_MODEL = "[^"]*"', "LLM_MODEL = `"$Model`""
                Set-Content rag\config.py $cfg -NoNewline
                Ok "rag\config.py now uses $Model"
            }
        }
        # Count the sections actually indexed: an empty rag\chroma_db folder is created
        # as soon as the backend checks RAG status, so the folder alone proves nothing.
        Push-Location rag
        $kb = & $VenvPy -c "from retrieve import get_collection; print(get_collection().count())" 2>$null
        Pop-Location
        $kbCount = 0
        if ($kb -and ("$kb".Trim() -match '^\d+$')) { $kbCount = [int]"$kb".Trim() }
        if ($kbCount -gt 0) { Ok "RAG knowledge base: $kbCount sections indexed" }
        else {
            Miss "RAG knowledge base is empty"; $Missing += "ingest"
            if (-not $CheckOnly) {
                Push-Location rag; & $VenvPy ingest.py; Pop-Location
            }
        }
    }
}

# ---------------------------------------------------------------- summary
Step "Summary"
if ($CheckOnly) {
    if ($Missing.Count -eq 0) { Ok "everything is ready. Run without -CheckOnly to start" }
    else { Miss ("missing: " + ($Missing -join ", ") + ". Run without -CheckOnly to install them") }
    exit 0
}

# ---------------------------------------------------------------- 6. run
Step "Starting backend on http://127.0.0.1:8000"
foreach ($port in 8000, 5173) {
    if (-not (Get-Command Get-NetTCPConnection -ErrorAction SilentlyContinue)) { break }
    $busy = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if ($busy) { Bad "port $port is already in use (an old run?). Close it first."; exit 1 }
}
$backend = Start-Process -FilePath $VenvPy -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000" `
    -WorkingDirectory (Join-Path $Root "backend") -NoNewWindow -PassThru
$up = $false
for ($i = 0; $i -lt 60 -and -not $up; $i++) {
    Start-Sleep 1
    if ($backend.HasExited) { break }
    try { Invoke-RestMethod "http://127.0.0.1:8000/health" -TimeoutSec 2 | Out-Null; $up = $true } catch { }
}
if (-not $up) { Bad "backend did not start (see errors above)"; if (-not $backend.HasExited) { Stop-Process -Id $backend.Id -Force }; exit 1 }
Ok "backend running"

Step "Starting dashboard: open http://localhost:5173 (admin / admin123). Ctrl+C stops everything."
try {
    Push-Location frontend
    npm run dev
}
finally {
    Pop-Location
    if (-not $backend.HasExited) { Stop-Process -Id $backend.Id -Force }
    Write-Host "`nStopped backend and dashboard." -ForegroundColor Cyan
}
