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

# Torch/torchvision first, from the pinned CUDA-11.8 wheel index.
# Installing these before nerfstudio keeps pip's
# resolver from silently swapping in a CPU-only or differently-CUDA'd build.
& $VenvPython -m pip install torch==2.1.2 torchvision==0.16.2 `
    --index-url https://download.pytorch.org/whl/cu118
if ($LASTEXITCODE -ne 0) { throw 'torch/torchvision install failed' }

& $VenvPython -m pip install `
    'git+https://github.com/nerfstudio-project/nerfstudio.git@758ea1918e082aa44776009d8e755c2f3a88d2ee'
if ($LASTEXITCODE -ne 0) { throw 'nerfstudio install failed' }

& $VenvPython (Join-Path $ProjectRoot 'scripts\apply_nerfstudio_patch.py')
if ($LASTEXITCODE -ne 0) { throw 'nerfstudio patch failed' }

& $VenvPython -m pip install --editable "$ProjectRoot[dev]"
if ($LASTEXITCODE -ne 0) { throw 'gsdb editable install failed' }

Write-Output 'Main environment installed. Install the shared trainer/evaluator with scripts\bootstrap-gsplat-windows.ps1, then run gsdb.ps1 doctor --backend all.'
