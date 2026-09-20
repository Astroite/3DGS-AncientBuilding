# Dot-source this file from any directory. Nothing is executed until Invoke-Gsdb.
param([string]$DataRoot)
$global:GsdbAppRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$global:GsdbPython = Join-Path $global:GsdbAppRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $global:GsdbPython -PathType Leaf)) {
    throw 'Native venv missing. Run APP/scripts/bootstrap-windows.ps1 first.'
}
if (-not $DataRoot) { $DataRoot = $env:GSDB_DATA_ROOT }
if (-not $DataRoot) { $DataRoot = Join-Path (Split-Path $global:GsdbAppRoot -Parent) 'Data' }
$env:GSDB_DATA_ROOT = (Resolve-Path -LiteralPath $DataRoot).Path
$env:PYTHONUNBUFFERED = '1'
# Force UTF-8 for child processes so Windows console code pages cannot
# corrupt tool output capture (GBK vs UTF-8 decode failures).
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
try {
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
    [Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
} catch {
    # Host may not expose a console; Python env vars above still apply.
}
$helper = Join-Path $global:GsdbAppRoot 'tools\mediasdk-helper\build\Release\gsdb-media-helper.exe'
if (-not $env:GSDB_MEDIA_HELPER -and (Test-Path -LiteralPath $helper)) {
    $env:GSDB_MEDIA_HELPER = $helper
}
$sdkRoot = Join-Path $global:GsdbAppRoot 'sdks\Insta360-MediaSDK'
if (-not $env:INSTA360_MEDIA_SDK_ROOT -and (Test-Path -LiteralPath (Join-Path $sdkRoot 'bin\MediaSDK.dll'))) {
    $env:INSTA360_MEDIA_SDK_ROOT = $sdkRoot
}
function global:Invoke-Gsdb {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$GsdbArguments)
    Push-Location -LiteralPath $global:GsdbAppRoot
    try {
        & $global:GsdbPython -m gsdb @GsdbArguments
        if ($LASTEXITCODE -ne 0) { throw "gsdb exited with code $LASTEXITCODE" }
    }
    finally { Pop-Location }
}
