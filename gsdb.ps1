[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$GsdbArgs
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$VenvPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    throw "GSDB virtual environment is missing: $VenvPython. Run scripts\bootstrap-windows.ps1 first."
}

& $VenvPython -m gsdb @GsdbArgs
exit $LASTEXITCODE
