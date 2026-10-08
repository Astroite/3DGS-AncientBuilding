param(
    [switch]$SkipPackages,
    [ValidatePattern('^14\.\d+(\.\d+)?$')]
    [string]$MsvcVersion
)
$ErrorActionPreference = 'Stop'
$AppRoot = Split-Path $PSScriptRoot -Parent
$Python = Join-Path $AppRoot '.venv-gsplat\Scripts\python.exe'

function Get-GsstudioCudaHome {
    foreach ($Candidate in @($env:GSSTUDIO_CUDA_HOME, $env:CUDA_HOME)) {
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

function Get-GsstudioVcvars {
    $Locator = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path -LiteralPath $Locator)) { return $null }
    $Install = & $Locator -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $Install) { return $null }
    $Vcvars = Join-Path $Install 'VC\Auxiliary\Build\vcvars64.bat'
    if (Test-Path -LiteralPath $Vcvars) { return $Vcvars }
    return $null
}

function Get-GsstudioCudaArch {
    if ($env:GSSTUDIO_CUDA_ARCH_LIST) {
        if ($env:GSSTUDIO_CUDA_ARCH_LIST -notmatch '^\d+\.\d+$') {
            throw 'GSSTUDIO_CUDA_ARCH_LIST must be a single compute capability, e.g. 8.9 or 12.0.'
        }
        return $env:GSSTUDIO_CUDA_ARCH_LIST
    }
    $Smi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    if (-not $Smi) { throw 'nvidia-smi is unavailable; set GSSTUDIO_CUDA_ARCH_LIST explicitly.' }
    $Capabilities = @(& $Smi.Source --query-gpu=compute_cap --format=csv,noheader)
    if ($LASTEXITCODE -ne 0 -or $Capabilities.Count -eq 0) {
        throw 'Cannot query GPU compute capability; set GSSTUDIO_CUDA_ARCH_LIST explicitly.'
    }
    $Unique = @($Capabilities | ForEach-Object { $_.Trim() } | Sort-Object -Unique)
    if ($Unique.Count -ne 1 -or $Unique[0] -notmatch '^\d+\.\d+$') {
        throw "Multiple or invalid GPU architectures ($($Unique -join ', ')); set GSSTUDIO_CUDA_ARCH_LIST explicitly."
    }
    return $Unique[0]
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
    # its CUDA extension is compiled for this machine's GPU by build_gsplat_csrc.py, which
    # caches the artifact under wheels/. See docs/MAINTENANCE.md.
    & $Python -m pip install gsplat==1.5.3 --no-deps
    if ($LASTEXITCODE -ne 0) { throw 'Cannot install the pinned gsplat sources' }

    $CudaHome = Get-GsstudioCudaHome
    if (-not $CudaHome) { throw 'CUDA 13 toolkit not found. Set GSSTUDIO_CUDA_HOME to its root.' }
    $Vcvars = Get-GsstudioVcvars
    if (-not $Vcvars) { throw 'MSVC build environment was not found.' }
    $EnvBat = Join-Path ([System.IO.Path]::GetTempPath()) ('gsstudio-vcvars-' + [guid]::NewGuid().ToString('N') + '.bat')
    $VcvarsArgument = if ($MsvcVersion) { " -vcvars_ver=$MsvcVersion" } else { '' }
    Set-Content -LiteralPath $EnvBat -Encoding ascii -Value ("@echo off`r`ncall `"$Vcvars`"$VcvarsArgument >nul`r`nif errorlevel 1 exit /b 1`r`nset`r`n")
    $BuildEnv = & cmd.exe /c $EnvBat
    $BuildEnvExit = $LASTEXITCODE
    Remove-Item -LiteralPath $EnvBat -Force
    if ($BuildEnvExit -ne 0) { throw 'Cannot initialize the selected MSVC build environment.' }
    foreach ($Line in $BuildEnv) {
        if ($Line -match '^([^=]+)=(.*)$') { Set-Item -Path ('env:' + $matches[1]) -Value $matches[2] -ErrorAction SilentlyContinue }
    }
    $env:CUDA_HOME = $CudaHome
    $env:CUDA_PATH = $CudaHome
    $env:PATH = (Join-Path $CudaHome 'bin') + [IO.Path]::PathSeparator + $env:PATH
    $env:GSSTUDIO_CUDA_ARCH_LIST = Get-GsstudioCudaArch
    $env:TORCH_CUDA_ARCH_LIST = $env:GSSTUDIO_CUDA_ARCH_LIST
    $env:MAX_JOBS = '4'

    & $Python (Join-Path $AppRoot 'scripts\build_gsplat_csrc.py')
    if ($LASTEXITCODE -ne 0) { throw "Cannot build the pinned CUDA extension for compute capability $env:GSSTUDIO_CUDA_ARCH_LIST" }

    & $Python -m pip install -e $AppRoot
    if ($LASTEXITCODE -ne 0) { throw 'Cannot install gsstudio training entrypoint' }

    # The Qt editor renders and picks Gaussians in its own process, so its
    # interpreter needs the same compiled gsplat extension as the trainer.
    $MainPython = Join-Path $AppRoot '.venv\Scripts\python.exe'
    $GpuAbi = (& $Python -c 'import sys,torch; print(f"{sys.version_info[0]}.{sys.version_info[1]}|{torch.__version__}")').Trim()
    $MainAbi = (& $MainPython -c 'import sys,torch; print(f"{sys.version_info[0]}.{sys.version_info[1]}|{torch.__version__}")').Trim()
    if ($GpuAbi -ne $MainAbi) { throw "Main and training Python/torch ABIs differ: $MainAbi vs $GpuAbi" }
    & $MainPython -m pip install gsplat==1.5.3 --no-deps
    if ($LASTEXITCODE -ne 0) { throw 'Cannot install gsplat Python package into main environment' }
    $GpuPackage = (& $Python -c 'import gsplat,pathlib; print(pathlib.Path(gsplat.__file__).parent)').Trim()
    $MainPackage = (& $MainPython -c 'import gsplat,pathlib; print(pathlib.Path(gsplat.__file__).parent)').Trim()
    foreach ($Name in @('csrc.pyd', 'csrc-build.json')) {
        Copy-Item -LiteralPath (Join-Path $GpuPackage $Name) -Destination (Join-Path $MainPackage $Name) -Force
    }
    & $MainPython -c 'from gsplat import csrc; print("Main Qt gsplat extension available")'
    if ($LASTEXITCODE -ne 0) { throw 'Main Qt environment cannot load the pinned gsplat extension' }
}
Write-Output "Native trainer: $Python"
