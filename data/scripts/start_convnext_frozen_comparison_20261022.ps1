param([switch]$Resume)
$ErrorActionPreference='Stop'
$cnnRoot=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$cnnOut=Join-Path $cnnRoot 'runs\experiments\convnext_frozen_comparison_20261022_v1'
$cnnTask='MarsKG-convnext-frozen-comparison-20261022-v1'
$cnnState=Get-Content -LiteralPath (Join-Path $cnnOut 'queue.json') -Raw | ConvertFrom-Json
$cnnExisting=Get-ScheduledTask -TaskName $cnnTask -ErrorAction SilentlyContinue
if ($cnnExisting -and $cnnExisting.State -eq 'Running') { throw 'Task already active' }
if (@(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*run_convnext_frozen_comparison_20261022.py*' }).Count) { throw 'Runner active' }
if ($Resume) {
    if ($cnnState.status -notin @('failed','paused_by_user')) { throw 'Expected stopped experiment' }
    $cnnMode='scheduled-resume'
} else {
    if ($cnnState.status -ne 'prepared_not_started' -or $cnnExisting) { throw 'Preserve existing experiment' }
    $cnnMode='scheduled-run'
}
$cnnAction=New-ScheduledTaskAction -Execute 'D:\Anaconda3\python.exe' -Argument ('-X utf8 -u "'+(Join-Path $PSScriptRoot 'run_convnext_frozen_comparison_20261022.py')+'" '+$cnnMode) -WorkingDirectory $cnnRoot
if ($Resume) { Set-ScheduledTask -TaskName $cnnTask -Action $cnnAction | Out-Null } else {
    $cnnSettings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 12) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    $cnnPrincipal=New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $cnnTask -Action $cnnAction -Settings $cnnSettings -Principal $cnnPrincipal -Description 'Finite frozen ConvNeXt versus DINOv2 native224 paired spatial comparison; offline' | Out-Null
}
Start-ScheduledTask -TaskName $cnnTask
Get-ScheduledTask -TaskName $cnnTask | Select-Object TaskName,State
