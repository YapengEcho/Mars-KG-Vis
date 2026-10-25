param([switch]$Resume)
$ErrorActionPreference='Stop'
$adaptRoot=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$adaptOut=Join-Path $adaptRoot 'runs\experiments\convnext_stage_adapt_20261022_v1'
$adaptTask='MarsKG-convnext-stage-adapt-20261022-v1'
$adaptState=Get-Content -LiteralPath (Join-Path $adaptOut 'queue.json') -Raw | ConvertFrom-Json
$adaptExisting=Get-ScheduledTask -TaskName $adaptTask -ErrorAction SilentlyContinue
if ($adaptExisting -and $adaptExisting.State -eq 'Running') { throw 'Task already active' }
if (@(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*run_convnext_stage_adapt_20261022.py*' }).Count) { throw 'Runner active' }
if ($Resume) {
    if ($adaptState.status -notin @('failed','paused_by_user')) { throw 'Expected stopped experiment' }
    $adaptMode='scheduled-resume'
} else {
    if ($adaptState.status -ne 'prepared_not_started' -or $adaptExisting) { throw 'Preserve existing experiment' }
    $adaptMode='scheduled-run'
}
$adaptAction=New-ScheduledTaskAction -Execute 'D:\Anaconda3\python.exe' -Argument ('-X utf8 -u "'+(Join-Path $PSScriptRoot 'run_convnext_stage_adapt_20261022.py')+'" '+$adaptMode) -WorkingDirectory $adaptRoot
if ($Resume) { Set-ScheduledTask -TaskName $adaptTask -Action $adaptAction | Out-Null } else {
    $adaptSettings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 12) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    $adaptPrincipal=New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $adaptTask -Action $adaptAction -Settings $adaptSettings -Principal $adaptPrincipal -Description 'Finite paired ConvNeXt head-only versus last-stage adaptation; offline; no automatic following jobs' | Out-Null
}
Start-ScheduledTask -TaskName $adaptTask
Get-ScheduledTask -TaskName $adaptTask | Select-Object TaskName,State
