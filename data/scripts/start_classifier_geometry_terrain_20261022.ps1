param([switch]$Resume)
$ErrorActionPreference='Stop'
$geometryRunRoot=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$geometryRunOut=Join-Path $geometryRunRoot 'runs\experiments\classifier_geometry_terrain_20261022_v1'
$geometryRunTask='MarsKG-classifier-geometry-terrain-20261022-v1'
$geometryRunState=Get-Content -LiteralPath (Join-Path $geometryRunOut 'queue.json') -Raw | ConvertFrom-Json
$geometryRunExisting=Get-ScheduledTask -TaskName $geometryRunTask -ErrorAction SilentlyContinue
if ($geometryRunExisting -and $geometryRunExisting.State -eq 'Running') { throw 'Task already running' }
if (@(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*run_classifier_geometry_terrain_20261022.py*' }).Count) { throw 'Runner active' }
if ($Resume) {
    if ($geometryRunState.status -notin @('failed','paused_by_user')) { throw 'Expected stopped experiment' }
    $geometryRunMode='scheduled-resume'
} else {
    if ($geometryRunState.status -ne 'prepared_not_started' -or $geometryRunExisting) { throw 'Preserve existing experiment' }
    $geometryRunMode='scheduled-run'
}
$geometryRunAction=New-ScheduledTaskAction -Execute 'D:\Anaconda3\python.exe' -Argument ('-X utf8 -u "'+(Join-Path $PSScriptRoot 'run_classifier_geometry_terrain_20261022.py')+'" '+$geometryRunMode) -WorkingDirectory $geometryRunRoot
if ($Resume) { Set-ScheduledTask -TaskName $geometryRunTask -Action $geometryRunAction | Out-Null } else {
    $geometryRunSettings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 12) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    $geometryRunPrincipal=New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $geometryRunTask -Action $geometryRunAction -Settings $geometryRunSettings -Principal $geometryRunPrincipal -Description 'Finite5arm fixed-LoRA geometry native-MOLA comparison; CPU2threads; no periodic trigger' | Out-Null
}
Start-ScheduledTask -TaskName $geometryRunTask
Get-ScheduledTask -TaskName $geometryRunTask | Select-Object TaskName,State
