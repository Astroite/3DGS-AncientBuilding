[CmdletBinding()]
param(
    [switch]$EnableVisionQa,
    [string]$Version = 'v001'
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$LocationId = 'yanguan-ancient-town-20260822'
$SceneId = 'night-walk-4k'
$CaptureId = 'capture-009-4k'
$InputVideo = Join-Path $ProjectRoot 'locations\yanguan-ancient-town-20260822\scenes\night-walk-4k\inputs\stitched\capture-009-4k-equirect.mp4'

if (-not (Test-Path -LiteralPath $InputVideo -PathType Leaf)) {
    throw "Studio export is missing: $InputVideo"
}

$WslPathInput = $ProjectRoot -replace '\\', '/'
$WslRoot = (& wsl.exe -d Ubuntu-22.04 -- wslpath -a $WslPathInput).Trim()
$WslHome = (& wsl.exe -d Ubuntu-22.04 -- sh -c 'printf %s "$HOME"').Trim()
$CondaExe = "$WslHome/miniforge3/bin/conda"
& wsl.exe -d Ubuntu-22.04 -- test -x $CondaExe
if ($LASTEXITCODE -ne 0) {
    throw 'Miniforge/Conda was not found in WSL. Run scripts/bootstrap-wsl.sh first.'
}

if ($EnableVisionQa) {
    if (-not $env:MIMO_API_KEY -or -not $env:MIMO_BASE_URL) {
        throw 'EnableVisionQa requires MIMO_API_KEY and MIMO_BASE_URL in this PowerShell session.'
    }
    $Forwarded = @('MIMO_API_KEY', 'MIMO_BASE_URL')
    $ExistingForwarded = @($env:WSLENV -split ':' | Where-Object { $_ })
    $env:WSLENV = (@($ExistingForwarded + $Forwarded) | Select-Object -Unique) -join ':'
}

function Invoke-GsdbStage {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    & wsl.exe -d Ubuntu-22.04 --cd $WslRoot -- $CondaExe run -n 3dgs --no-capture-output gsdb @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "gsdb stage failed: $($Arguments -join ' ')"
    }
}

Push-Location $ProjectRoot
try {
    Invoke-GsdbStage -Arguments @('doctor')
    Invoke-GsdbStage -Arguments @('ingest', $LocationId, $SceneId, $CaptureId, '--resume')

    $VisionArgument = if ($EnableVisionQa) { '--vision-qa' } else { '--no-vision-qa' }
    $PreprocessOutput = & wsl.exe -d Ubuntu-22.04 --cd $WslRoot -- $CondaExe run -n 3dgs --no-capture-output gsdb preprocess $LocationId $SceneId $CaptureId $VisionArgument 2>&1
    $PreprocessOutput | ForEach-Object { Write-Host $_ }
    if ($LASTEXITCODE -ne 0) {
        throw 'gsdb preprocess failed'
    }
    $RunLine = $PreprocessOutput | Select-String -Pattern '^RUN_ID=(.+)$' | Select-Object -Last 1
    if (-not $RunLine) {
        throw 'Could not obtain RUN_ID from gsdb preprocess output'
    }
    $RunId = $RunLine.Matches[0].Groups[1].Value.Trim()
    Write-Host "Continuing masked demo run $RunId"

    Invoke-GsdbStage -Arguments @('mask', $LocationId, $SceneId, $RunId)
    Invoke-GsdbStage -Arguments @('reconstruct', $LocationId, $SceneId, $RunId)
    Invoke-GsdbStage -Arguments @('train', $LocationId, $SceneId, $RunId)
    Invoke-GsdbStage -Arguments @('export', $LocationId, $SceneId, $RunId, '--version', $Version)
    Invoke-GsdbStage -Arguments @('qa', 'report', $LocationId, $SceneId, $RunId)
    Invoke-GsdbStage -Arguments @('catalog', 'build')

    Write-Host "Completed $RunId. Artifacts remain needs_review; no automatic approval was performed."
}
finally {
    Pop-Location
}
