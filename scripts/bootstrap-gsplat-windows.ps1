param([switch]$SkipPackages)
$ErrorActionPreference = 'Stop'
$AppRoot = Split-Path $PSScriptRoot -Parent
$Python = Join-Path $AppRoot '.venv-gsplat\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) {
    & (Join-Path $AppRoot '.venv\Scripts\python.exe') -m venv (Join-Path $AppRoot '.venv-gsplat')
    if ($LASTEXITCODE -ne 0) { throw 'Cannot create native training environment' }
}
if (-not $SkipPackages) {
    & $Python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
    if ($LASTEXITCODE -ne 0) { throw 'Cannot install pinned CUDA PyTorch' }
    & $Python -m pip install numpy==1.26.4 scipy==1.14.1 ninja==1.11.1.3 rich==13.9.4 lpips==0.1.4 jaxtyping
    if ($LASTEXITCODE -ne 0) { throw 'Cannot install pinned training dependencies' }
    & $Python -m pip install gsplat==1.4.0+pt21cu118 --no-deps --index-url https://docs.gsplat.studio/whl/pt21cu118
    if ($LASTEXITCODE -ne 0) { throw 'Cannot install pinned Windows CUDA extension' }
    & $Python -m pip install -e $AppRoot
    if ($LASTEXITCODE -ne 0) { throw 'Cannot install gsdb training entrypoint' }
}
Write-Output "Native trainer: $Python"
