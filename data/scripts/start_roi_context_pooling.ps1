$ErrorActionPreference = 'Stop'
$poolRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$poolOut = Join-Path $poolRoot 'runs\experiments\roi_context_pooling_20261015_v1'
$poolTask = 'MarsKG-roi-context-pooling-20261015-v1'
if ((Get-Content -LiteralPath (Join-Path $poolOut 'queue.json') -Raw | ConvertFrom-Json).status -ne 'prepared_not_started') { throw 'Preserve existing experiment' }
if ((Get-FileHash -LiteralPath (Join-Path $poolOut 'protocol.json')).Hash -ne (Get-Content -LiteralPath (Join-Path $poolOut 'protocol.sha256') -Raw).Trim()) { throw 'Protocol changed' }
if (Get-ScheduledTask -TaskName $poolTask -ErrorAction SilentlyContinue) { throw 'Task exists' }
$poolArgs = '-u "' + (Join-Path $poolRoot 'data\scripts\run_roi_context_pooling.py') + '" scheduled-run'
$poolAction = New-ScheduledTaskAction -Execute 'D:\Anaconda3\python.exe' -Argument $poolArgs -WorkingDirectory $poolRoot
$poolSettings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 1) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$poolPrincipal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $poolTask -Action $poolAction -Settings $poolSettings -Principal $poolPrincipal -Description 'Offline frozen DINOv2 unchanged full-image ROI/context pooling comparison' | Select-Object TaskName,State
Start-ScheduledTask -TaskName $poolTask
Get-ScheduledTask -TaskName $poolTask | Select-Object TaskName,State
