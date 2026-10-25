param([switch]$Resume)
$ErrorActionPreference='Stop'
$repeatRoot=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$repeatOut=Join-Path $repeatRoot 'runs\experiments\detector_recipe_repeat_20261021_v1'
$repeatTaskName='MarsKG-detector-recipe-repeat-20261021-v1'
$repeatState=Get-Content -LiteralPath (Join-Path $repeatOut 'queue.json') -Raw | ConvertFrom-Json
$repeatExistingTask=Get-ScheduledTask -TaskName $repeatTaskName -ErrorAction SilentlyContinue
if ($repeatExistingTask -and $repeatExistingTask.State -eq 'Running') { throw 'Task already running' }
$repeatProcesses=@(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*run_detector_recipe_repeat_20261021.py*' })
if ($repeatProcesses.Count -gt 0) { throw 'Training process already running' }
if ($Resume) {
    if ($repeatState.status -ne 'paused_by_user') { throw 'Expected an explicitly paused experiment' }
    foreach ($repeatCheckpoint in $repeatState.checkpoints_at_pause) {
        if ((Get-FileHash -LiteralPath (Join-Path $repeatOut $repeatCheckpoint.path)).Hash.ToLower() -ne $repeatCheckpoint.sha256) { throw 'Paused checkpoint changed' }
    }
    $repeatHistory=Join-Path $repeatOut ('resume_history\'+(Get-Date -Format 'yyyyMMdd_HHmmss'))
    New-Item -ItemType Directory -Path $repeatHistory | Out-Null
    Copy-Item -LiteralPath (Join-Path $repeatOut 'queue.json') -Destination (Join-Path $repeatHistory 'queue.before_resume.json')
    $repeatMode='scheduled-resume'
} else {
    if ($repeatState.status -ne 'prepared_not_started' -or $repeatExistingTask) { throw 'Preserve existing experiment' }
    $repeatMode='scheduled-run'
}
$repeatProtocolPath=Join-Path $repeatOut 'protocol.json'
if ((Get-FileHash -LiteralPath $repeatProtocolPath).Hash.ToLower() -ne (Get-Content -LiteralPath (Join-Path $repeatOut 'protocol.sha256') -Raw).Trim()) { throw 'Protocol changed' }
$repeatProtocol=Get-Content -LiteralPath $repeatProtocolPath -Raw | ConvertFrom-Json
foreach ($repeatPin in $repeatProtocol.files_sha256.PSObject.Properties) {
    if ((Get-FileHash -LiteralPath $repeatPin.Name).Hash.ToLower() -ne $repeatPin.Value) { throw "Pinned file changed: $($repeatPin.Name)" }
}
$repeatAction=New-ScheduledTaskAction -Execute 'D:\Anaconda3\python.exe' -Argument ('-X utf8 -u "'+(Join-Path $PSScriptRoot 'run_detector_recipe_repeat_20261021.py')+'" '+$repeatMode) -WorkingDirectory $repeatRoot
if ($Resume) {
    Set-ScheduledTask -TaskName $repeatTaskName -Action $repeatAction | Out-Null
} else {
    $repeatSettings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 24) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    $repeatPrincipal=New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $repeatTaskName -Action $repeatAction -Settings $repeatSettings -Principal $repeatPrincipal -Description 'Finite seed1 nbs64/nbs8 paired detector repeat; short test, training and common evaluation; no periodic trigger' | Out-Null
}
Start-ScheduledTask -TaskName $repeatTaskName
Get-ScheduledTask -TaskName $repeatTaskName | Select-Object TaskName,State
