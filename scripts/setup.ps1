# One-time setup for this workspace (safe to re-run; finished steps are skipped or re-verified).
#   1. create .venv (Python 3.11) and install packages with uv
#   2. download the two public QuantiPhy datasets + submission template into data\
#   3. fetch the official starter kit (evaluator.py) at a pinned commit into external\QuantiPhy
#
# Usage (from the project folder):
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$StarterRepo = "https://github.com/Paulineli/QuantiPhy"
$StarterCommit = "4f9323c9ca9479fc673749ae7d2a82729fef6e85"   # 2026-09-16, pinned
$StarterDir = Join-Path $ProjectRoot "external\QuantiPhy"

function Fail($msg) { Write-Host "SETUP FAILED: $msg" -ForegroundColor Red; exit 1 }

$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) { $uv = "$env:USERPROFILE\.local\bin\uv.exe" }
if (-not (Test-Path $uv)) { Fail "uv not found (expected on PATH or at $uv)" }

Write-Host "[1/3] uv sync (Python 3.11 venv in .venv)"
& $uv sync
if ($LASTEXITCODE -ne 0) { Fail "uv sync" }

Write-Host "[2/3] download datasets (pinned revisions) -> data\"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
& $uv run python scripts\download_data.py
if ($LASTEXITCODE -ne 0) { Fail "download_data.py" }

Write-Host "[3/3] starter kit $StarterCommit -> external\QuantiPhy"
$have = $null
if (Test-Path (Join-Path $StarterDir ".git")) { $have = (git -C $StarterDir rev-parse HEAD 2>$null) }
if ($have -eq $StarterCommit) {
    Write-Host "      already at pinned commit"
} else {
    if (Test-Path $StarterDir) { Fail "$StarterDir exists but is not at $StarterCommit; move it away and re-run" }
    New-Item -ItemType Directory -Force $StarterDir | Out-Null
    git -C $StarterDir init -q
    git -C $StarterDir remote add origin $StarterRepo
    git -C $StarterDir fetch -q --depth 1 origin $StarterCommit
    if ($LASTEXITCODE -ne 0) { Fail "git fetch of starter kit" }
    git -C $StarterDir checkout -q FETCH_HEAD
    if ($LASTEXITCODE -ne 0) { Fail "git checkout of starter kit" }
}
$have = (git -C $StarterDir rev-parse HEAD 2>$null)
if ($have -ne $StarterCommit) { Fail "starter kit is at $have, expected $StarterCommit" }

Write-Host "SETUP OK" -ForegroundColor Green
