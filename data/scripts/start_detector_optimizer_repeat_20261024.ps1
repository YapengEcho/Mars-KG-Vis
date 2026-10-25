param([switch]$Resume)
$ErrorActionPreference='Stop'
$repeatRoot=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$repeatOut=Join-Path $repeatRoot 'runs\experiments\detector_optimizer_repeat_20261024_v1'
$repeatTask='MarsKG-detector-optimizer-repeat-20261024-v1'
$repeatState=Get-Content -LiteralPath (Join-Path $repeatOut 'queue.json') -Raw | ConvertFrom-Json
$existing=Get-ScheduledTask -TaskName $repeatTask -ErrorAction SilentlyContinue
if ($existing -and $existing.State -eq 'Running') { throw 'Repeat already active' }
if ($Resume) {
    if ($repeatState.status -notin @('failed_resumable','paused_by_user')) { throw 'Expected stopped repeat' }
    $mode='scheduled-resume'
} else {
    if ($existing -or $repeatState.status -ne 'prepared_not_started') { throw 'Preserve existing experiment' }
    $mode='scheduled-run'
}
$action=New-ScheduledTaskAction -Execute 'D:\Anaconda3\python.exe' -Argument ('-X utf8 -u "'+(Join-Path $PSScriptRoot 'run_detector_optimizer_repeat_20261024.py')+'" '+$mode) -WorkingDirectory $repeatRoot
if ($existing) { Set-ScheduledTask -TaskName $repeatTask -Action $action | Out-Null } else {
    $settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 24) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    $principal=New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $repeatTask -Action $action -Settings $settings -Principal $principal -Description 'Finite second detector seed: paired AdamW3e-4 versus SGD1e-2; GPU smoke, early stopping, same frozen manual cohort; no periodic triggers' | Out-Null
}
Start-ScheduledTask -TaskName $repeatTask
Get-ScheduledTask -TaskName $repeatTask | Select-Object TaskName,State
