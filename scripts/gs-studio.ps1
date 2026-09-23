# Launch the GS-Studio workflow and model workbench.
param([string]$DataRoot)
$gsAppRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$candidates = @()
if ($env:GSSTUDIO_GSPLAT_PYTHON) { $candidates += $env:GSSTUDIO_GSPLAT_PYTHON }
$candidates += (Join-Path $gsAppRoot '.venv-gsplat\Scripts\python.exe')
$candidates += (Join-Path $gsAppRoot '.venv\Scripts\python.exe')
$gsPython = $candidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
if (-not $gsPython) {
    throw 'Python environment missing. Run scripts/bootstrap-windows.ps1 first (or bootstrap-gsplat-windows.ps1 for the training environment).'
}
& $gsPython -c 'import PySide6' 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "PySide6 is missing in $gsPython. Install the studio extra first: pip install -e `"$gsAppRoot[studio]`""
}
if ($DataRoot) {
    $env:GSSTUDIO_DATA_ROOT = (Resolve-Path -LiteralPath $DataRoot).Path
}
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
Push-Location -LiteralPath $gsAppRoot
try {
    & $gsPython -m gsstudio.interfaces.desktop
    if ($LASTEXITCODE -ne 0) { throw "GS-Studio exited with code $LASTEXITCODE" }
}
finally { Pop-Location }
