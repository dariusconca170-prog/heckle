# Download the model candidates into data/models.
# Verify repos/file names before relying on them: community uploads get renamed.
param(
    [ValidateSet("9b", "35b", "4b", "all")]
    [string[]]$Model = @("9b"),
    [string]$ModelsDir = "data/models"
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null

$python = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }
if (-not (Get-Command hf -ErrorAction SilentlyContinue)) {
    Write-Host "installing huggingface_hub CLI ..."
    & $python -m pip install "huggingface_hub[cli]"
    $hf = Join-Path $Root ".venv\Scripts\hf.exe"
} else {
    $hf = "hf"
}

$wanted = if ($Model -contains "all") { @("9b", "35b", "4b") } else { $Model }
foreach ($name in $wanted) {
    switch ($name) {
        "9b" {
            Write-Host "== 9B default candidate ==" -ForegroundColor Cyan
            & $hf download HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive `
                --include "*Q4_K_M*" --include "*mmproj*" --local-dir $ModelsDir
        }
        "35b" {
            Write-Host "== 35B challenger ==" -ForegroundColor Cyan
            & $hf download HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive `
                --include "*Q4_K_P*" --include "*mmproj*" --local-dir $ModelsDir
        }
        "4b" {
            Write-Host "== 4B fit fallback ==" -ForegroundColor Cyan
            & $hf download prithivMLmods/Qwen3.5-abliterated-MAX-AIO-GGUF `
                --include "*4B*Q4*" --include "*4B*mmproj*" --local-dir $ModelsDir
        }
    }
}

Write-Host ""
Write-Host "Downloaded GGUF files:" -ForegroundColor Green
Get-ChildItem -Path $ModelsDir -Recurse -Filter *.gguf | ForEach-Object {
    $hash = (Get-FileHash $_.FullName -Algorithm SHA256).Hash
    "{0,10:N1} MB  {1}`n            sha256 {2}" -f ($_.Length / 1MB), $_.FullName, $hash
}
Write-Host ""
Write-Host "Check that config/models/<profile>.toml points at these exact filenames."
Write-Host "Do not start M1 until data/bakeoff.md names a winner."
