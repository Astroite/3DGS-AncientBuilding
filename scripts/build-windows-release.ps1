[CmdletBinding()]
param(
    [string]$MainPython,
    [string]$GpuPython,
    [string]$FfmpegDir,
    [string]$OutputRoot
)

$ErrorActionPreference = 'Stop'
$AppRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if (-not $MainPython) { $MainPython = Join-Path $AppRoot '.venv\Scripts\python.exe' }
if (-not $GpuPython) { $GpuPython = Join-Path $AppRoot '.venv-gsplat\Scripts\python.exe' }
if (-not $OutputRoot) { $OutputRoot = Join-Path $AppRoot 'dist' }
$MainPython = [IO.Path]::GetFullPath($MainPython)
$GpuPython = [IO.Path]::GetFullPath($GpuPython)
$OutputRoot = [IO.Path]::GetFullPath($OutputRoot)
$Smi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if (-not $Smi) { throw 'Cannot identify target GPU; nvidia-smi.exe is unavailable.' }
$Capabilities = @(& $Smi.Source --query-gpu=compute_cap --format=csv,noheader)
if ($LASTEXITCODE -ne 0 -or $Capabilities.Count -eq 0) { throw 'Cannot query target GPU compute capability.' }
$UniqueCapabilities = @($Capabilities | ForEach-Object { $_.Trim() } | Sort-Object -Unique)
if ($UniqueCapabilities.Count -ne 1 -or $UniqueCapabilities[0] -notmatch '^\d+\.\d+$') {
    throw "Target GPUs have mixed or invalid capabilities: $($UniqueCapabilities -join ', ')"
}
$GpuArch = $UniqueCapabilities[0]
$env:GSSTUDIO_CUDA_ARCH_LIST = $GpuArch
if (-not (Test-Path -LiteralPath $MainPython -PathType Leaf)) { throw "Main Python is missing: $MainPython" }
if (-not (Test-Path -LiteralPath $GpuPython -PathType Leaf)) { throw "Pinned GPU Python is missing: $GpuPython. Run scripts/bootstrap-gsplat-windows.ps1 first." }
if (-not $FfmpegDir) { throw 'Pass -FfmpegDir with ffmpeg.exe, ffprobe.exe and their redistribution license.' }
$FfmpegDir = [IO.Path]::GetFullPath($FfmpegDir)
$FfmpegBinaryDir = if (Test-Path -LiteralPath (Join-Path $FfmpegDir 'ffmpeg.exe') -PathType Leaf) {
    $FfmpegDir
} elseif (Test-Path -LiteralPath (Join-Path $FfmpegDir 'bin\ffmpeg.exe') -PathType Leaf) {
    Join-Path $FfmpegDir 'bin'
} else {
    throw "FFmpeg binary directory is missing under $FfmpegDir"
}
foreach ($Name in @('ffmpeg.exe', 'ffprobe.exe')) {
    if (-not (Test-Path -LiteralPath (Join-Path $FfmpegBinaryDir $Name) -PathType Leaf)) {
        throw "Bundled $Name is missing from $FfmpegBinaryDir"
    }
}
$FfmpegParent = Split-Path $FfmpegBinaryDir -Parent
$Licenses = @(
    @(Get-ChildItem -LiteralPath $FfmpegBinaryDir -File -ErrorAction SilentlyContinue) +
    @(Get-ChildItem -LiteralPath $FfmpegParent -File -ErrorAction SilentlyContinue) |
    Where-Object { $_.Name -match '^(LICENSE|COPYING)' } |
    Sort-Object FullName -Unique
)
if ($Licenses.Count -eq 0) { throw "FFmpeg redistribution license is missing beside $FfmpegBinaryDir" }

$VersionText = Get-Content -LiteralPath (Join-Path $AppRoot 'pyproject.toml') -Raw -Encoding utf8
$VersionMatch = [regex]::Match($VersionText, '(?m)^version\s*=\s*"([^"]+)"')
if (-not $VersionMatch.Success) { throw 'Cannot read project version from pyproject.toml' }
$Version = $VersionMatch.Groups[1].Value
$GitCommit = (& git -C $AppRoot rev-parse --verify HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Cannot read Git commit' }
$SourceState = if ((& git -C $AppRoot status --porcelain).Count -gt 0) { 'modified' } else { 'clean' }

$env:PYTHONPATH = Join-Path $AppRoot 'src'
$CheckMain = @'
import sys
import platform
from importlib.metadata import version
for package in ('PyInstaller', 'PySide6', 'Pillow', 'torch', 'torchvision', 'scipy', 'opencv-python-headless', 'gsplat', 'gsstudio'):
    print(package, version(package))
import torch, gsplat, pathlib, json, hashlib, os
from gsplat import csrc
assert version('PyInstaller') == '6.16.0', 'Expected pinned PyInstaller 6.16.0'
assert torch.__version__.split('+')[0] == '2.9.1' and torch.version.cuda == '13.0', 'Main runtime must use torch 2.9.1+cu130'
assert version('gsplat').split('+')[0] == '1.5.3', 'Qt editor requires gsplat 1.5.3'
assert sys.version_info[:2] in ((3, 10), (3, 11), (3, 12)), 'Unsupported Python ABI'
assert platform.machine().lower() in ('amd64', 'x86_64'), 'Windows x64 build required'
root = pathlib.Path(gsplat.__file__).parent
extension = root / 'csrc.pyd'
record = json.loads((root / 'csrc-build.json').read_text(encoding='utf-8'))
assert record['compute_capability'] == os.environ['GSSTUDIO_CUDA_ARCH_LIST'], 'Qt gsplat extension targets another GPU architecture'
assert record['torch'] == torch.__version__, 'Qt gsplat extension targets another torch ABI'
assert record['extension_sha256'] == hashlib.sha256(extension.read_bytes()).hexdigest(), 'Qt gsplat extension identity changed'
'@
& $MainPython -c $CheckMain
if ($LASTEXITCODE -ne 0) { throw 'Main packaging environment is incomplete or unpinned' }
$CheckGpu = @'
import sys
from importlib.metadata import version
for package in ('PyInstaller', 'torch', 'torchvision', 'gsplat', 'scipy', 'lpips', 'gsstudio'):
    print(package, version(package))
import torch
from gsplat import csrc
import gsplat, pathlib, json, hashlib, os
assert version('PyInstaller') == '6.16.0', 'Expected pinned PyInstaller 6.16.0'
assert torch.__version__.split('+')[0] == '2.9.1' and torch.version.cuda == '13.0', 'GPU runtime must use torch 2.9.1+cu130'
assert version('gsplat').split('+')[0] == '1.5.3', 'GPU runtime must use gsplat 1.5.3'
assert sys.version_info[:2] in ((3, 10), (3, 11), (3, 12)), 'Unsupported Python ABI'
root = pathlib.Path(gsplat.__file__).parent
extension = root / 'csrc.pyd'
record = json.loads((root / 'csrc-build.json').read_text(encoding='utf-8'))
assert record['compute_capability'] == os.environ['GSSTUDIO_CUDA_ARCH_LIST'], 'gsplat extension was built for another GPU architecture'
assert record['torch'] == torch.__version__, 'gsplat extension was built for another torch ABI'
assert record['extension_sha256'] == hashlib.sha256(extension.read_bytes()).hexdigest(), 'gsplat extension identity changed'
'@
& $GpuPython -c $CheckGpu
if ($LASTEXITCODE -ne 0) { throw 'GPU packaging environment is incomplete or unpinned' }

$BuildId = Get-Date -Format 'yyyyMMdd-HHmmss'
$WorkRoot = Join-Path $OutputRoot ("build-$BuildId")
$ReleaseRoot = Join-Path $OutputRoot ("GSStudio-$Version-win64-cuda13-$BuildId")
if ((Test-Path -LiteralPath $WorkRoot) -or (Test-Path -LiteralPath $ReleaseRoot)) {
    throw "Build output already exists: $BuildId"
}
New-Item -ItemType Directory -Path $WorkRoot, $ReleaseRoot | Out-Null
$SpecDir = Join-Path $WorkRoot 'spec'
$WorkDir = Join-Path $WorkRoot 'work'
$DistDir = Join-Path $WorkRoot 'dist'
$ResourceDir = Join-Path $AppRoot 'src\gsstudio\resources'

function Invoke-Freeze {
    param([string]$Python, [string]$Name, [string]$Entry, [switch]$Windowed,
          [string[]]$CollectAll = @(), [string[]]$Metadata = @(),
          [string[]]$ExtraData = @())
    $Arguments = @(
        '-m', 'PyInstaller', '--noconfirm', '--clean', '--onedir',
        '--name', $Name, '--specpath', $SpecDir, '--workpath', $WorkDir,
        '--distpath', $DistDir, '--paths', (Join-Path $AppRoot 'src'),
        '--add-data', "$ResourceDir;gsstudio/resources"
    )
    if ($Windowed) { $Arguments += '--windowed' } else { $Arguments += '--console' }
    foreach ($Package in $CollectAll) { $Arguments += @('--collect-all', $Package) }
    foreach ($Package in $Metadata) { $Arguments += @('--copy-metadata', $Package) }
    foreach ($Datum in $ExtraData) { $Arguments += @('--add-data', $Datum) }
    $Arguments += $Entry
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed while building $Name" }
    $Result = Join-Path $DistDir $Name
    if (-not (Test-Path -LiteralPath (Join-Path $Result "$Name.exe") -PathType Leaf)) {
        throw "PyInstaller did not create $Name.exe"
    }
    return $Result
}

$MainCsrcMetadata = (& $MainPython -c "import gsplat,pathlib; print(pathlib.Path(gsplat.__file__).parent / 'csrc-build.json')").Trim()
if ($LASTEXITCODE -ne 0) { throw 'Cannot locate Qt gsplat CUDA build identity' }
$Desktop = Invoke-Freeze -Python $MainPython -Name 'GSStudio' `
    -Entry (Join-Path $AppRoot 'scripts\frozen_desktop.py') -Windowed `
    -CollectAll @('torch', 'torchvision', 'gsplat', 'cv2') `
    -Metadata @('torch', 'torchvision', 'gsplat', 'opencv-python-headless', 'gsstudio') `
    -ExtraData @("$MainCsrcMetadata;gsplat")
$Worker = Invoke-Freeze -Python $MainPython -Name 'GSStudioWorker' `
    -Entry (Join-Path $AppRoot 'scripts\frozen_worker.py') `
    -CollectAll @('torch', 'torchvision', 'gsplat', 'cv2') `
    -Metadata @('torch', 'torchvision', 'gsplat', 'opencv-python-headless', 'gsstudio') `
    -ExtraData @("$MainCsrcMetadata;gsplat")
$Cli = Invoke-Freeze -Python $MainPython -Name 'GSStudioCLI' `
    -Entry (Join-Path $AppRoot 'scripts\frozen_cli.py') `
    -CollectAll @('torch', 'torchvision', 'gsplat', 'cv2') `
    -Metadata @('torch', 'torchvision', 'gsplat', 'opencv-python-headless', 'gsstudio') `
    -ExtraData @("$MainCsrcMetadata;gsplat")
$GpuCsrcMetadata = (& $GpuPython -c "import gsplat,pathlib; print(pathlib.Path(gsplat.__file__).parent / 'csrc-build.json')").Trim()
if ($LASTEXITCODE -ne 0) { throw 'Cannot locate gsplat CUDA build identity' }
$Trainer = Invoke-Freeze -Python $GpuPython -Name 'GSStudioTrainer' `
    -Entry (Join-Path $AppRoot 'scripts\frozen_trainer.py') `
    -CollectAll @('torch', 'torchvision', 'gsplat', 'lpips', 'cv2') `
    -Metadata @('torch', 'torchvision', 'gsplat', 'opencv-python-headless', 'gsstudio') `
    -ExtraData @("$GpuCsrcMetadata;gsplat")

Get-ChildItem -LiteralPath $Desktop -Force |
    Copy-Item -Destination $ReleaseRoot -Recurse -Force
foreach ($Pair in @(@('worker', $Worker), @('cli', $Cli), @('gpu-runtime', $Trainer))) {
    $Destination = Join-Path $ReleaseRoot $Pair[0]
    New-Item -ItemType Directory -Path $Destination | Out-Null
    Copy-Item -Path (Join-Path $Pair[1] '*') -Destination $Destination -Recurse
}
$FfmpegDestination = Join-Path $ReleaseRoot 'tools\ffmpeg'
New-Item -ItemType Directory -Path $FfmpegDestination -Force | Out-Null
foreach ($Name in @('ffmpeg.exe', 'ffprobe.exe')) {
    Copy-Item -LiteralPath (Join-Path $FfmpegBinaryDir $Name) -Destination $FfmpegDestination
}
Get-ChildItem -LiteralPath $FfmpegBinaryDir -File -Filter '*.dll' |
    Copy-Item -Destination $FfmpegDestination
$Licenses | Copy-Item -Destination $FfmpegDestination
$RealityscanDestination = Join-Path $ReleaseRoot 'tools\realityscan-setup'
New-Item -ItemType Directory -Path $RealityscanDestination -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $AppRoot 'tools\realityscan-setup\colmap-export-params.xml') `
    -Destination $RealityscanDestination
foreach ($Name in @('verify-release.ps1', 'diagnostics.ps1')) {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot $Name) -Destination $ReleaseRoot
}
Copy-Item -LiteralPath (Join-Path $AppRoot 'docs\development\WINDOWS-RELEASE.md') `
    -Destination (Join-Path $ReleaseRoot 'README-RELEASE.md')
& $MainPython (Join-Path $PSScriptRoot 'write_release_manifest.py') $ReleaseRoot `
    --version $Version --commit $GitCommit --source-state $SourceState `
    --gpu-compute-capability $GpuArch
if ($LASTEXITCODE -ne 0) { throw 'Cannot write release manifest' }
& (Join-Path $ReleaseRoot 'verify-release.ps1')
if ($LASTEXITCODE -ne 0) { throw 'Release verification failed' }
Write-Output "GS Studio release assembled: $ReleaseRoot"
