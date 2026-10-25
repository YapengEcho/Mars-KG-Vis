param([switch]$Resume)
$ErrorActionPreference='Stop'
$bridgeRoot=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$bridgeOut=Join-Path $bridgeRoot 'runs\experiments\convnext_dino_bridge_20261024_v1'
$bridgeTask='MarsKG-convnext-dino-bridge-20261024-v1'
$bridgeState=Get-Content -LiteralPath (Join-Path $bridgeOut 'queue.json') -Raw | ConvertFrom-Json
$bridgeExisting=Get-ScheduledTask -TaskName $bridgeTask -ErrorAction SilentlyContinue
if ($bridgeExisting -and $bridgeExisting.State -eq 'Running') { throw 'Task already active' }
if (@(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*run_convnext_dino_bridge_20261024.py*' }).Count) { throw 'Runner active' }
if ($Resume) {
    if ($bridgeState.status -notin @('failed','paused_by_user')) { throw 'Expected stopped experiment' }
    $bridgeMode='scheduled-resume'
} else {
    if ($bridgeState.status -ne 'prepared_not_started' -or $bridgeExisting) { throw 'Preserve existing experiment' }
    $bridgeMode='scheduled-run'
}
$bridgeAction=New-ScheduledTaskAction -Execute 'D:\Anaconda3\python.exe' -Argument ('-X utf8 -u "'+(Join-Path $PSScriptRoot 'run_convnext_dino_bridge_20261024.py')+'" '+$bridgeMode) -WorkingDirectory $bridgeRoot
if ($Resume) { Set-ScheduledTask -TaskName $bridgeTask -Action $bridgeAction | Out-Null } else {
    $bridgeSettings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 12) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    $bridgePrincipal=New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $bridgeTask -Action $bridgeAction -Settings $bridgeSettings -Principal $bridgePrincipal -Description 'Finite three fixed-epoch ConvNeXt fits and same-pool detector classification ensemble; offline; no following jobs' | Out-Null
}
Start-ScheduledTask -TaskName $bridgeTask
Get-ScheduledTask -TaskName $bridgeTask | Select-Object TaskName,State
