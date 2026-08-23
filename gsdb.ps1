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
    'MIMO_API_KEY',
    'MIMO_BASE_URL'
) | Where-Object { Test-Path -LiteralPath "Env:$_" }
if ($ForwardedNames.Count -gt 0) {
    $ExistingForwarded = @($env:WSLENV -split ':' | Where-Object { $_ })
    $env:WSLENV = (@($ExistingForwarded + $ForwardedNames) | Select-Object -Unique) -join ':'
}

& wsl.exe -d Ubuntu-22.04 --cd $WslRoot -- $CondaExe run -n 3dgs --no-capture-output gsdb @GsdbArgs
exit $LASTEXITCODE
