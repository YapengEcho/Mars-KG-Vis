param([switch]$Resume)
$ErrorActionPreference='Stop'
$cachedHeadRoot=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$cachedHeadOut=Join-Path $cachedHeadRoot 'runs\experiments\classifier_cached_heads_20261022_v1'
$cachedHeadTask='MarsKG-classifier-cached-heads-20261022-v1'
$cachedHeadState=Get-Content -LiteralPath (Join-Path $cachedHeadOut 'queue.json') -Raw | ConvertFrom-Json
$cachedHeadExisting=Get-ScheduledTask -TaskName $cachedHeadTask -ErrorAction SilentlyContinue
if ($cachedHeadExisting -and $cachedHeadExisting.State -eq 'Running') { throw 'Task already running' }
if (@(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*run_classifier_cached_heads_20261022.py*' }).Count) { throw 'Runner already active' }
if ($Resume) {
    if ($cachedHeadState.status -notin @('failed','paused_by_user')) { throw 'Expected stopped experiment' }
    $cachedHeadMode='scheduled-resume'
} else {
    if ($cachedHeadState.status -ne 'prepared_not_started' -or $cachedHeadExisting) { throw 'Preserve existing experiment' }
    $cachedHeadMode='scheduled-run'
}
$cachedHeadAction=New-ScheduledTaskAction -Execute 'D:\Anaconda3\python.exe' -Argument ('-X utf8 -u "'+(Join-Path $PSScriptRoot 'run_classifier_cached_heads_20261022.py')+'" '+$cachedHeadMode) -WorkingDirectory $cachedHeadRoot
if ($Resume) {
    Set-ScheduledTask -TaskName $cachedHeadTask -Action $cachedHeadAction | Out-Null
} else {
    $cachedHeadSettings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 12) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    $cachedHeadPrincipal=New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $cachedHeadTask -Action $cachedHeadAction -Settings $cachedHeadSettings -Principal $cachedHeadPrincipal -Description 'Finite phase-aligned classifier head comparison; offline; no periodic trigger' | Out-Null
}
Start-ScheduledTask -TaskName $cachedHeadTask
Get-ScheduledTask -TaskName $cachedHeadTask | Select-Object TaskName,State
