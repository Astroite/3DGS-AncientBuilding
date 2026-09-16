# Launch the offline training GUI with the existing dedicated GPU environment.
param([string]$Dataset, [string]$Project)
$studioAppRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$studioPython = if ($env:GSDB_GSPLAT_PYTHON) { $env:GSDB_GSPLAT_PYTHON } else { Join-Path $studioAppRoot '.venv-gsplat\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $studioPython -PathType Leaf)) {
    throw 'Training environment missing. Run scripts/bootstrap-gsplat-windows.ps1 first.'
}
$studioArgs = @('-m', 'gsdb.studio')
if ($Dataset) { $studioArgs += @('--dataset', $Dataset) }
if ($Project) { $studioArgs += @('--project', $Project) }
Push-Location -LiteralPath $studioAppRoot
try {
    & $studioPython @studioArgs
    if ($LASTEXITCODE -ne 0) { throw "Studio exited with code $LASTEXITCODE" }
}
finally { Pop-Location }
