[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$GsdbArgs
)

$ErrorActionPreference = 'Stop'
try {
    # session.ps1 selects .venv and preserves configured SDK/Data locations.
    . (Join-Path $PSScriptRoot 'scripts\session.ps1')
    Invoke-Gsdb @GsdbArgs
    exit 0
}
catch {
    Write-Error -Message $_ -ErrorAction Continue
    exit 1
}
