param([Parameter(Mandatory=$true)][string]$Receipt)
$ErrorActionPreference='Stop'
$workspace=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path.TrimEnd('\')
$receiptRoot=Join-Path $workspace 'runs\maintenance\catalog_expansion_20261024_v1'
$receiptPath=(Resolve-Path -LiteralPath $Receipt).Path
if (-not $receiptPath.StartsWith($receiptRoot+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Unexpected deletion receipt' }
$plan=Get-Content -LiteralPath $receiptPath -Raw -Encoding utf8 | ConvertFrom-Json
if ($plan.status -eq 'deleted_verified_original') { return }
if ($plan.status -ne 'verified_native_regions_preserved') { throw 'Native verification not complete' }
$allowedRoots=@((Join-Path $workspace 'data\raw\ctx_tiles'),(Join-Path $env:TEMP 'MarsKG-catalog-expansion-20261024\ctx'))
$target=(Resolve-Path -LiteralPath $plan.original.path).Path
$inside=$false
foreach ($root in $allowedRoots) { if ($target.StartsWith([IO.Path]::GetFullPath($root).TrimEnd('\')+'\',[StringComparison]::OrdinalIgnoreCase)) { $inside=$true } }
if (-not $inside) { throw 'Original outside explicit CTX roots' }
$item=Get-Item -LiteralPath $target
if ($item.PSIsContainer -or $item.LinkType -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'Unexpected target kind' }
if ($item.Name -ne ('MurrayLab_CTX_V01_'+$plan.tile+'_Mosaic.tif')) { throw 'Unexpected original filename' }
if ($item.Length -ne $plan.original.bytes -or (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -ne $plan.original.sha256) { throw 'Original changed' }
$cacheRoot=Join-Path $workspace 'data\processed\catalog_native_regions_20261024_v1'
if ((Get-FileHash -LiteralPath $plan.cache_manifest -Algorithm SHA256).Hash -ne $plan.cache_manifest_sha256) { throw 'Cache manifest changed' }
foreach ($cache in $plan.caches) {
    $p=(Resolve-Path -LiteralPath $cache.path).Path
    if (-not $p.StartsWith($cacheRoot+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Cache outside designated directory' }
    $c=Get-Item -LiteralPath $p
    if ($c.PSIsContainer -or $c.LinkType -or $c.Length -ne $cache.bytes -or (Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash -ne $cache.sha256) { throw 'Preserved cache changed' }
    if ($cache.decoded_pixels_verified -ne $true) { throw 'Decoded native pixels not verified' }
}
if (@($plan.caches).Count -eq 0 -or $plan.required_windows_covered -ne $true) { throw 'Preservation coverage incomplete' }
Remove-Item -LiteralPath $target -Force
if (Test-Path -LiteralPath $target) { throw 'Deletion failed' }
$plan.status='deleted_verified_original'
$plan | Add-Member -NotePropertyName deleted_utc -NotePropertyValue ([DateTime]::UtcNow.ToString('o')) -Force
$plan | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $receiptPath -Encoding utf8
[pscustomobject]@{tile=$plan.tile;removed_bytes=$item.Length;cache_bytes=($plan.caches | Measure-Object bytes -Sum).Sum}
