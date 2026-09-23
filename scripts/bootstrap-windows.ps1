[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path -LiteralPath "$PSScriptRoot\..").Path
$VenvPath = Join-Path $ProjectRoot '.venv'
$VenvPython = Join-Path $VenvPath 'Scripts\python.exe'

if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    $SystemPython = Get-Command python -ErrorAction SilentlyContinue
    if (-not $SystemPython) {
        throw 'No system Python found on PATH. Install Python 3.10 first.'
    }
    & $SystemPython.Source -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) { throw 'Python environment creation failed' }
}

& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed' }

# Torch/torchvision first, from the CUDA 13 wheel index.
# Installing these before nerfstudio keeps pip's
# resolver from silently swapping in a CPU-only or differently-CUDA'd build.
# sm_120 (Blackwell) hosts cannot execute CUDA 11.8 builds.
& $VenvPython -m pip install torch==2.9.1 torchvision==0.24.1 `
    --index-url https://download.pytorch.org/whl/cu130
if ($LASTEXITCODE -ne 0) { throw 'torch/torchvision install failed' }

# nerfstudio stays on the pinned commit, but --no-deps keeps pip from building the
# CUDA 11.8-only extensions it pins (gsplat 1.4.0, nerfacc 0.5.2). GSSTUDIO only imports
# nerfstudio's COLMAP/equirect helpers; the subset below is their complete
# third-party import surface.
& $VenvPython -m pip install --no-deps `
    'git+https://github.com/nerfstudio-project/nerfstudio.git@758ea1918e082aa44776009d8e755c2f3a88d2ee'
if ($LASTEXITCODE -ne 0) { throw 'nerfstudio install failed' }

& $VenvPython -m pip install `
    appdirs 'opencv-python-headless==4.10.0.84' numpy requests packaging rich matplotlib jaxtyping imageio rawpy
if ($LASTEXITCODE -ne 0) { throw 'nerfstudio runtime dependency install failed' }

& $VenvPython (Join-Path $ProjectRoot 'scripts\apply_nerfstudio_patch.py')
if ($LASTEXITCODE -ne 0) { throw 'nerfstudio patch failed' }

& $VenvPython -m pip install --editable "$ProjectRoot[dev]"
if ($LASTEXITCODE -ne 0) { throw 'gsstudio editable install failed' }

Write-Output 'Main environment installed. Install the shared trainer/evaluator with scripts\bootstrap-gsplat-windows.ps1, then run gsstudio.ps1 doctor --backend gsplat.'
