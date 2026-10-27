param([switch]$Resume)
$ErrorActionPreference='Stop'
$repeatRoot=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$repeatOut=Join-Path $repeatRoot 'runs\experiments\convnext_stage_repeat_20261023_v1'
$repeatTask='MarsKG-convnext-stage-repeat-20261023-v1'
$repeatState=Get-Content -LiteralPath (Join-Path $repeatOut 'queue.json') -Raw | ConvertFrom-Json
$repeatExisting=Get-ScheduledTask -TaskName $repeatTask -ErrorAction SilentlyContinue
if ($repeatExisting -and $repeatExisting.State -eq 'Running') { throw 'Task already active' }
if (@(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*run_convnext_stage_repeat_20261023.py*' }).Count) { throw 'Runner active' }
if ($Resume) {
    if ($repeatState.status -notin @('failed','paused_by_user')) { throw 'Expected stopped experiment' }
    $repeatMode='scheduled-resume'
} else {
    if ($repeatState.status -ne 'prepared_not_started' -or $repeatExisting) { throw 'Preserve existing experiment' }
    $repeatMode='scheduled-run'
}
$repeatAction=New-ScheduledTaskAction -Execute 'D:\Anaconda3\python.exe' -Argument ('-X utf8 -u "'+(Join-Path $PSScriptRoot 'run_convnext_stage_repeat_20261023.py')+'" '+$repeatMode) -WorkingDirectory $repeatRoot
if ($Resume) { Set-ScheduledTask -TaskName $repeatTask -Action $repeatAction | Out-Null } else {
    $repeatSettings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 12) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    $repeatPrincipal=New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $repeatTask -Action $repeatAction -Settings $repeatSettings -Principal $repeatPrincipal -Description 'Finite two paired ConvNeXt repeat seeds; offline; no following jobs' | Out-Null
}
Start-ScheduledTask -TaskName $repeatTask
Get-ScheduledTask -TaskName $repeatTask | Select-Object TaskName,State
