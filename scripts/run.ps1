# Start the UI (and, if selected, the model sidecar).
param(
    [int]$Port = 8000,
    [switch]$NoBrowser,
    [switch]$Reload,
    [switch]$WithModel
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$venvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "No .venv found. Run .\scripts\setup.ps1 first."
}
if (-not (Test-Path "frontend\dist\index.html")) {
    Push-Location frontend
    try { npm run build } finally { Pop-Location }
}

$modelProcess = $null
try {
    if ($WithModel -or (Test-Path "data\model_profile.json")) {
        if (Test-Path "data\model_profile.json") {
            Write-Host "starting llama-server from data\model_profile.json ..."
            $modelProcess = Start-Process -FilePath $venvPython `
                -ArgumentList @("scripts\serve_model.py", "data\model_profile.json") `
                -PassThru -NoNewWindow
        } else {
            Write-Warning "-WithModel requested but data\model_profile.json does not exist; run the M0.5 bake-off first."
        }
    }

    $url = "http://127.0.0.1:$Port/"
    if (-not $NoBrowser) {
        Start-Process $url | Out-Null
    }
    $uvicornArgs = @("-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "$Port")
    if ($Reload) { $uvicornArgs += "--reload" }
    & $venvPython @uvicornArgs
} finally {
    if ($modelProcess -and -not $modelProcess.HasExited) {
        Stop-Process -Id $modelProcess.Id -Force -ErrorAction SilentlyContinue
    }
}
