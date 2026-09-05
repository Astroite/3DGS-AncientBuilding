[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$LocationId,
    [Parameter(Mandatory = $true)][string]$SceneId,
    [Parameter(Mandatory = $true)][string]$CaptureId,
    [ValidateRange(180, 5000)][int]$TargetFrames = 270,
    [switch]$ResumePreparedCache
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if (-not $env:GSDB_MEDIA_HELPER) {
    throw 'Set GSDB_MEDIA_HELPER to the approved Windows x64 helper first.'
}
$ConfiguredHelper = Get-Item -LiteralPath $env:GSDB_MEDIA_HELPER
if ($ConfiguredHelper.Extension -ne '.exe') {
    throw 'Hardware acceptance requires the approved Windows x64 .exe, not the CI fake helper.'
}
$HelperStream = [System.IO.File]::OpenRead($ConfiguredHelper.FullName)
$HelperReader = [System.IO.BinaryReader]::new($HelperStream)
try {
    if ($HelperReader.ReadUInt16() -ne 0x5A4D) {
        throw 'Hardware acceptance requires a valid Windows PE executable.'
    }
    $HelperStream.Position = 0x3C
    $PeOffset = $HelperReader.ReadInt32()
    $HelperStream.Position = $PeOffset
    if ($HelperReader.ReadUInt32() -ne 0x00004550 -or $HelperReader.ReadUInt16() -ne 0x8664) {
        throw 'Hardware acceptance requires a Windows x64 PE executable.'
    }
}
finally {
    $HelperReader.Dispose()
    $HelperStream.Dispose()
}
if (-not $env:INSTA360_MEDIA_SDK_ROOT) {
    throw 'Set INSTA360_MEDIA_SDK_ROOT to the approved SDK installation first.'
}
$Gsdb = Join-Path $ProjectRoot 'gsdb.ps1'
& $Gsdb doctor --require mediasdk
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host 'Probing the real INSV source...'
& $Gsdb media probe $LocationId $SceneId $CaptureId
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$PreprocessArguments = @(
    'preprocess', $LocationId, $SceneId, $CaptureId,
    '--target-frames', [string]$TargetFrames
)
if ($ResumePreparedCache) {
    $PreprocessArguments += '--resume'
}

Write-Host "Exporting and validating $TargetFrames selected frames..."
$PreprocessOutput = @(& $Gsdb @PreprocessArguments 2>&1)
$PreprocessOutput | ForEach-Object { Write-Host $_ }
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$RunLines = @($PreprocessOutput | ForEach-Object { [string]$_ } |
    Where-Object { $_ -match '^RUN_ID=([a-zA-Z0-9_-]+)$' })
if ($RunLines.Count -ne 1) {
    throw 'Could not determine the run ID from preprocess output.'
}
$RunId = [regex]::Match($RunLines[0], '^RUN_ID=(.+)$').Groups[1].Value
$SceneRoot = Join-Path $ProjectRoot "locations\$LocationId\scenes\$SceneId"
$RunManifestPath = Join-Path $SceneRoot "runs\$RunId.yaml"
$PreparedMatch = Select-String -LiteralPath $RunManifestPath `
    -Pattern '^\s*prepared_relative_path:\s*(.+?)\s*$'
if ($PreparedMatch.Count -ne 1) {
    throw "Run manifest does not contain one prepared_relative_path: $RunManifestPath"
}
$PreparedRelative = $PreparedMatch.Matches[0].Groups[1].Value.Trim()
$PreparedRelative = $PreparedRelative.Trim([char[]]@([char]39, [char]34))
$DatasetDirectory = [System.IO.Path]::GetFullPath(
    (Join-Path $SceneRoot $PreparedRelative.Replace('/', '\'))
)
$ScenePrefix = ([System.IO.Path]::GetFullPath($SceneRoot)).TrimEnd('\') + '\'
if (-not $DatasetDirectory.StartsWith($ScenePrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Run prepared path escapes the scene: $PreparedRelative"
}
$DatasetManifestPath = Join-Path $DatasetDirectory 'dataset.json'
if (-not (Test-Path -LiteralPath $DatasetManifestPath -PathType Leaf)) {
    throw "Prepared input manifest is missing: $DatasetManifestPath"
}

$Manifest = Get-Content -Raw -LiteralPath $DatasetManifestPath |
    ConvertFrom-Json -Depth 100
if (
    $Manifest.integrity -ne 'complete' -or
    $Manifest.lineage.capture_id -ne $CaptureId -or
    [int]$Manifest.lineage.target_frames -ne $TargetFrames -or
    $Manifest.lineage.source_kind -ne 'insta360_insv'
) {
    throw "The run references an incompatible prepared-input manifest: $DatasetManifestPath"
}

$Frames = @($Manifest.frames)
if ($Frames.Count -ne $TargetFrames) {
    throw "Expected $TargetFrames frame records, found $($Frames.Count)."
}
$SourceFiles = @($Manifest.lineage.source_files)
if ($SourceFiles.Count -notin @(1, 2)) {
    throw "INSV lineage must contain one or two explicit source files."
}
$UniqueIndices = @($Frames | ForEach-Object { [int]$_.source_frame_index } |
    Sort-Object -Unique)
if ($UniqueIndices.Count -ne $TargetFrames) {
    throw 'Candidate source-frame indices are not unique.'
}
if (-not $Manifest.lineage.helper_version -or -not $Manifest.lineage.sdk_version) {
    throw 'The dataset does not record both helper_version and sdk_version.'
}
if (
    [string]$Manifest.lineage.helper_version -match '^fake-' -or
    [string]$Manifest.lineage.sdk_version -match '^fake-'
) {
    throw 'Hardware acceptance refuses fake helper or SDK version metadata.'
}

try {
    Add-Type -AssemblyName System.Drawing.Common -ErrorAction Stop
}
catch {
    Add-Type -AssemblyName System.Drawing
}

$DatasetPrefix = $DatasetDirectory.TrimEnd('\') + '\'
foreach ($Frame in $Frames) {
    $RelativeFrame = ([string]$Frame.file).Replace('/', '\')
    $FramePath = [System.IO.Path]::GetFullPath((Join-Path $DatasetDirectory $RelativeFrame))
    if (-not $FramePath.StartsWith($DatasetPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Frame path escapes the dataset: $RelativeFrame"
    }
    if (-not (Test-Path -LiteralPath $FramePath -PathType Leaf)) {
        throw "Frame is missing: $FramePath"
    }
    $Digest = (Get-FileHash -Algorithm SHA256 -LiteralPath $FramePath).Hash.ToLowerInvariant()
    if ($Digest -ne [string]$Frame.sha256) {
        throw "Frame hash mismatch: $FramePath"
    }
    $Image = [System.Drawing.Image]::FromFile($FramePath)
    try {
        if (
            $Image.Width -ne [int]$Frame.width -or
            $Image.Height -ne [int]$Frame.height -or
            $Image.Width -ne 2 * $Image.Height
        ) {
            throw "Frame dimensions are invalid: $FramePath ($($Image.Width)x$($Image.Height))"
        }
    }
    finally {
        $Image.Dispose()
    }
}

[pscustomobject]@{
    status = 'passed'
    run_id = $RunId
    dataset = $DatasetDirectory
    source_file_count = $SourceFiles.Count
    frame_count = $Frames.Count
    width = [int]$Frames[0].width
    height = [int]$Frames[0].height
    helper_version = [string]$Manifest.lineage.helper_version
    sdk_version = [string]$Manifest.lineage.sdk_version
    preparation_hash = [string]$Manifest.preparation_hash
    dataset_sha256 = [string]$Manifest.dataset_sha256
} | ConvertTo-Json -Depth 4
