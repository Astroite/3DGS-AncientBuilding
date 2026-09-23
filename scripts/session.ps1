# Dot-source this file from any directory. Nothing is executed until Invoke-Gsstudio.
param([string]$DataRoot)
$global:GsstudioAppRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$global:GsstudioPython = Join-Path $global:GsstudioAppRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $global:GsstudioPython -PathType Leaf)) {
    throw 'Native venv missing. Run GSStudio/scripts/bootstrap-windows.ps1 first.'
}
if (-not $DataRoot) { $DataRoot = $env:GSSTUDIO_DATA_ROOT }
if (-not $DataRoot) { $DataRoot = Join-Path (Split-Path $global:GsstudioAppRoot -Parent) 'Data' }
$env:GSSTUDIO_DATA_ROOT = (Resolve-Path -LiteralPath $DataRoot).Path
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
$helper = Join-Path $global:GsstudioAppRoot 'tools\mediasdk-helper\build\Release\gsstudio-media-helper.exe'
if (-not $env:GSSTUDIO_MEDIA_HELPER -and (Test-Path -LiteralPath $helper)) {
    $env:GSSTUDIO_MEDIA_HELPER = $helper
}
$sdkRoot = Join-Path $global:GsstudioAppRoot 'sdks\Insta360-MediaSDK'
if (-not $env:INSTA360_MEDIA_SDK_ROOT -and (Test-Path -LiteralPath (Join-Path $sdkRoot 'bin\MediaSDK.dll'))) {
    $env:INSTA360_MEDIA_SDK_ROOT = $sdkRoot
}
function global:Invoke-Gsstudio {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$GsstudioArguments)
    Push-Location -LiteralPath $global:GsstudioAppRoot
    try {
        & $global:GsstudioPython -m gsstudio @GsstudioArguments
        if ($LASTEXITCODE -ne 0) { throw "gsstudio exited with code $LASTEXITCODE" }
    }
    finally { Pop-Location }
}
