# Windows setup for the twitch simulator (M0).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "== Twitch Simulator setup ==" -ForegroundColor Magenta

# --- Python 3.12 + venv -----------------------------------------------------
$py = Get-Command py -ErrorAction SilentlyContinue
if ($py) {
    & py -3.12 --version
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    & python --version
} else {
    throw "Python 3.12 is required. Install it from python.org and re-run."
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "creating .venv ..."
    if ($py) { & py -3.12 -m venv .venv } else { & python -m venv .venv }
}
$venvPython = Join-Path $Root ".venv\Scripts\python.exe"
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -e ".[dev]"

# --- Frontend ---------------------------------------------------------------
if (-not $env:TWITCHSIM_SKIP_FRONTEND) {
    if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
        throw "Node.js/npm is required to build the UI. Install Node 20+ and re-run."
    }
    Push-Location frontend
    try {
        if (Test-Path "package-lock.json") { npm ci } else { npm install }
        npm run build
    } finally {
        Pop-Location
    }
}

# --- Demo clip + external tools --------------------------------------------
& $venvPython scripts/make_sample.py

$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
if ($ffmpeg) {
    Write-Host "ffmpeg: $(& ffmpeg -version | Select-Object -First 1)"
} else {
    Write-Warning "ffmpeg is not on PATH. Video playback in the browser may work, but perception/ffmpeg features need it."
}
$nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidia) {
    & nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
} else {
    Write-Warning "nvidia-smi not found. M0.5 cannot measure VRAM."
}

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Run:  .\scripts\run.ps1"
Write-Host "Gate: .\scripts\fetch_models.ps1 -Model 9b, then read README.md (M0.5)."
