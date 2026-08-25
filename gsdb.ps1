[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$GsdbArgs
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$WslPathInput = $ProjectRoot -replace '\\', '/'
$WslRoot = (& wsl.exe -d Ubuntu-22.04 -- wslpath -a $WslPathInput).Trim()
if (-not $WslRoot) {
    throw 'Unable to translate the project path into WSL.'
}

$WslHome = (& wsl.exe -d Ubuntu-22.04 -- sh -c 'printf %s "$HOME"').Trim()
$CondaExe = "$WslHome/miniforge3/bin/conda"
& wsl.exe -d Ubuntu-22.04 -- test -x $CondaExe
if ($LASTEXITCODE -ne 0) {
    throw 'Miniforge/Conda was not found in WSL. Run scripts/bootstrap-wsl.sh first.'
}

$ForwardedNames = @(
    'DEEPSEEK_API_KEY'
) | Where-Object { Test-Path -LiteralPath "Env:$_" }
if ($ForwardedNames.Count -gt 0) {
    $ExistingForwarded = @($env:WSLENV -split ':' | Where-Object { $_ })
    $env:WSLENV = (@($ExistingForwarded + $ForwardedNames) | Select-Object -Unique) -join ':'
}

# Run scratch data goes on the WSL ext4 disk instead of the 9p /mnt/d bridge, where
# every image read costs about four times as much. Manifests and published artifacts
# stay in the project tree; only the intermediate bytes move.
$ScratchRoot = if ($env:GSDB_SCRATCH_ROOT) { $env:GSDB_SCRATCH_ROOT } else { "$WslHome/gsdb-scratch" }

& wsl.exe -d Ubuntu-22.04 --cd $WslRoot -- env "GSDB_SCRATCH_ROOT=$ScratchRoot" $CondaExe run -n 3dgs --no-capture-output gsdb @GsdbArgs
exit $LASTEXITCODE
