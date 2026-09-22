param([switch]$SkipPackages)
$ErrorActionPreference = 'Stop'
$AppRoot = Split-Path $PSScriptRoot -Parent
$Python = Join-Path $AppRoot '.venv-gsplat\Scripts\python.exe'

function Get-GsdbCudaHome {
    foreach ($Candidate in @($env:GSDB_CUDA_HOME, $env:CUDA_HOME)) {
        if ($Candidate -and (Test-Path -LiteralPath (Join-Path $Candidate 'bin\nvcc.exe'))) { return $Candidate }
    }
    $ToolkitRoot = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'
    if (Test-Path -LiteralPath $ToolkitRoot) {
        $Newest = Get-ChildItem -LiteralPath $ToolkitRoot -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -like 'v13.*' } | Sort-Object Name -Descending | Select-Object -First 1
        if ($Newest) { return $Newest.FullName }
    }
    $Local = Join-Path (Split-Path $AppRoot -Parent) 'tools\cuda-13.4'
    if (Test-Path -LiteralPath (Join-Path $Local 'bin\nvcc.exe')) { return $Local }
    return $null
}

function Get-GsdbVcvars {
    $Locator = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path -LiteralPath $Locator)) { return $null }
    $Install = & $Locator -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $Install) { return $null }
    $Vcvars = Join-Path $Install 'VC\Auxiliary\Build\vcvars64.bat'
    if (Test-Path -LiteralPath $Vcvars) { return $Vcvars }
    return $null
}

if (-not (Test-Path -LiteralPath $Python)) {
    & (Join-Path $AppRoot '.venv\Scripts\python.exe') -m venv (Join-Path $AppRoot '.venv-gsplat')
    if ($LASTEXITCODE -ne 0) { throw 'Cannot create native training environment' }
}
if (-not $SkipPackages) {
    & $Python -m pip install --upgrade pip setuptools wheel
    if ($LASTEXITCODE -ne 0) { throw 'Cannot upgrade packaging tools' }
    & $Python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130
    if ($LASTEXITCODE -ne 0) { throw 'Cannot install pinned CUDA PyTorch' }
    & $Python -m pip install numpy==1.26.4 scipy==1.14.1 ninja==1.11.1.3 rich==13.9.4 lpips==0.1.4 jaxtyping
    if ($LASTEXITCODE -ne 0) { throw 'Cannot install pinned training dependencies' }

    # gsplat publishes no cu130 wheel, so the package is installed as sources and
    # its CUDA extension is compiled once for sm_120 by build_gsplat_csrc.py, which
    # caches the artifact under wheels/. See docs/MAINTENANCE.md.
    & $Python -m pip install gsplat==1.5.3 --no-deps
    if ($LASTEXITCODE -ne 0) { throw 'Cannot install the pinned gsplat sources' }

    $CudaHome = Get-GsdbCudaHome
    if (-not $CudaHome) { throw 'CUDA 13 toolkit not found. Set GSDB_CUDA_HOME to its root.' }
    $Vcvars = Get-GsdbVcvars
    if (-not $Vcvars) { throw 'MSVC build environment was not found.' }
    $EnvBat = Join-Path ([System.IO.Path]::GetTempPath()) ('gsdb-vcvars-' + [guid]::NewGuid().ToString('N') + '.bat')
    Set-Content -LiteralPath $EnvBat -Encoding ascii -Value ("@echo off`r`ncall `"$Vcvars`" >nul`r`nset`r`n")
    $BuildEnv = & cmd.exe /c $EnvBat
    Remove-Item -LiteralPath $EnvBat -Force
    foreach ($Line in $BuildEnv) {
        if ($Line -match '^([^=]+)=(.*)$') { Set-Item -Path ('env:' + $matches[1]) -Value $matches[2] -ErrorAction SilentlyContinue }
    }
    $env:CUDA_HOME = $CudaHome
    $env:CUDA_PATH = $CudaHome
    $env:PATH = (Join-Path $CudaHome 'bin') + [IO.Path]::PathSeparator + $env:PATH
    $env:TORCH_CUDA_ARCH_LIST = '12.0'
    $env:MAX_JOBS = '4'

    & $Python (Join-Path $AppRoot 'scripts\build_gsplat_csrc.py')
    if ($LASTEXITCODE -ne 0) { throw 'Cannot build the pinned CUDA extension for sm_120' }

    & $Python -m pip install -e $AppRoot
    if ($LASTEXITCODE -ne 0) { throw 'Cannot install gsdb training entrypoint' }
}
Write-Output "Native trainer: $Python"
