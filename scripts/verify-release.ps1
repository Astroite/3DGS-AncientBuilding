[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$ReleaseRoot = Split-Path $MyInvocation.MyCommand.Path -Parent
$ManifestPath = Join-Path $ReleaseRoot 'release-manifest.json'
if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) {
    throw "Release manifest is missing: $ManifestPath"
}
$Manifest = Get-Content -LiteralPath $ManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
if ($Manifest.schema_version -ne 1 -or $Manifest.product -ne 'GS Studio') {
    throw 'Unsupported GS Studio release manifest'
}
$Base = [IO.Path]::GetFullPath($ReleaseRoot).TrimEnd('\') + '\'
$Expected = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
foreach ($File in $Manifest.files) {
    $Relative = [string]$File.path
    if ($Relative -eq 'release-manifest.json' -or -not $Expected.Add($Relative)) {
        throw "Duplicate or invalid manifest entry: $Relative"
    }
    $Target = [IO.Path]::GetFullPath((Join-Path $ReleaseRoot $Relative))
    if (-not $Target.StartsWith($Base, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Unsafe release path: $Relative"
    }
    if (-not (Test-Path -LiteralPath $Target -PathType Leaf)) {
        throw "Missing release file: $Relative"
    }
    $Info = Get-Item -LiteralPath $Target
    if ($Info.Length -ne [long]$File.bytes) {
        throw "Release file length changed: $Relative"
    }
    $Actual = (Get-FileHash -LiteralPath $Target -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Actual -ne [string]$File.sha256) {
        throw "Release file hash changed: $Relative"
    }
}
foreach ($File in Get-ChildItem -LiteralPath $ReleaseRoot -File -Recurse -Force) {
    if ($File.FullName -eq $ManifestPath) { continue }
    $Relative = $File.FullName.Substring($Base.Length).Replace('\', '/')
    if (-not $Expected.Contains($Relative)) {
        throw "Unexpected release file: $Relative"
    }
}
Write-Output "GS Studio $($Manifest.version): $($Manifest.files.Count) files verified."
