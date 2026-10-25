param([ValidateSet('Launch', 'Status')][string]$Mode = 'Launch')
$ErrorActionPreference = 'Stop'
$nativeComparisonRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$nativeComparisonOut = Join-Path $nativeComparisonRoot 'runs\experiments\native_crop_comparison_20261015_v1'
$nativeComparisonTask = 'MarsKG-native-crop-compare-20261015-v1'
if ($Mode -eq 'Status') {
    Get-ScheduledTask -TaskName $nativeComparisonTask | Select-Object TaskName, State
    Get-ScheduledTaskInfo -TaskName $nativeComparisonTask | Format-List LastRunTime, LastTaskResult
    Get-Content -LiteralPath (Join-Path $nativeComparisonOut 'queue.json')
    return
}
$nativeComparisonProtocolPath = Join-Path $nativeComparisonOut 'protocol.json'
$nativeComparisonExpectedHash = (Get-Content -LiteralPath (Join-Path $nativeComparisonOut 'protocol.sha256') -Raw).Trim()
if ((Get-FileHash -LiteralPath $nativeComparisonProtocolPath -Algorithm SHA256).Hash -ne $nativeComparisonExpectedHash) {
    throw 'Prepared comparison protocol hash changed'
}
$nativeComparisonQueue = Get-Content -LiteralPath (Join-Path $nativeComparisonOut 'queue.json') -Raw | ConvertFrom-Json
if ($nativeComparisonQueue.status -ne 'prepared_not_started') {
    throw 'Preserve existing run; launch only the reviewed prepared comparison'
}
$nativeComparisonExistingTask = Get-ScheduledTask -TaskName $nativeComparisonTask -ErrorAction SilentlyContinue
if ($nativeComparisonExistingTask) {
    throw 'A task with this exact name already exists; no silent replacement'
}
$nativeComparisonPython = 'D:\Anaconda3\python.exe'
if (-not (Test-Path -LiteralPath $nativeComparisonPython -PathType Leaf)) {
    throw 'Independent base Python executable unavailable'
}
$nativeComparisonRunner = Join-Path $nativeComparisonRoot 'data\scripts\run_native_crop_comparison.py'
$nativeComparisonArguments = '-u "' + $nativeComparisonRunner + '" scheduled-run --out "' + $nativeComparisonOut + '"'
$nativeComparisonAction = New-ScheduledTaskAction -Execute $nativeComparisonPython -Argument $nativeComparisonArguments -WorkingDirectory $nativeComparisonRoot
$nativeComparisonSettings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 6) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$nativeComparisonPrincipal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $nativeComparisonTask -Action $nativeComparisonAction -Settings $nativeComparisonSettings -Principal $nativeComparisonPrincipal -Description 'Offline full native CTX crop build and paired frozen DINOv2 classifier comparison; no client or network dependency' | Select-Object TaskName, State
Start-ScheduledTask -TaskName $nativeComparisonTask
Get-ScheduledTask -TaskName $nativeComparisonTask | Select-Object TaskName, State
Get-ScheduledTaskInfo -TaskName $nativeComparisonTask | Format-List LastRunTime, LastTaskResult
