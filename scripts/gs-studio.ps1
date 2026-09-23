# Launch the GS-Studio GUI shell (read-only project browsing, status and logs).
param([string]$DataRoot)
$gsAppRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$candidates = @()
if ($env:GSDB_GSPLAT_PYTHON) { $candidates += $env:GSDB_GSPLAT_PYTHON }
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
    $env:GSDB_DATA_ROOT = (Resolve-Path -LiteralPath $DataRoot).Path
}
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
Push-Location -LiteralPath $gsAppRoot
try {
    & $gsPython -m gsdb.gui
    if ($LASTEXITCODE -ne 0) { throw "GS-Studio exited with code $LASTEXITCODE" }
}
finally { Pop-Location }
