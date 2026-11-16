# One-time setup on Windows. Run from the repo root:
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -r backend\requirements.txt
.\.venv\Scripts\python.exe -m federated.data --clients 5
.\.venv\Scripts\python.exe -m federated.simulate --quiet
Push-Location frontend; npm install; Pop-Location
if (Get-Command ollama -ErrorAction SilentlyContinue) {
  ollama pull nomic-embed-text
  ollama pull llama3.2:3b
  Push-Location rag; ..\.venv\Scripts\python.exe ingest.py; Pop-Location
} else {
  Write-Host "Ollama not found. Install it from https://ollama.com/download for reports + chat."
}
Write-Host "Done. In VS Code: Run and Debug > 'Full system (backend + dashboard)', then open http://localhost:5173"
