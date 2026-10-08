[CmdletBinding()]
param([string]$DataRoot)

$ErrorActionPreference = 'Stop'
$ReleaseRoot = Split-Path $MyInvocation.MyCommand.Path -Parent
& (Join-Path $ReleaseRoot 'verify-release.ps1')
if ($DataRoot) { $env:GSSTUDIO_DATA_ROOT = (Resolve-Path -LiteralPath $DataRoot).Path }
& (Join-Path $ReleaseRoot 'cli\GSStudioCLI.exe') doctor --backend gsplat
if ($LASTEXITCODE -ne 0) { throw "GS Studio doctor failed with exit code $LASTEXITCODE" }
