$ErrorActionPreference = 'Stop'
$bridgeRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$bridgeOut = Join-Path $bridgeRoot 'runs\experiments\detector_classifier_bridge_20261015_v2'
$bridgeTask = 'MarsKG-detector-classifier-bridge-20261015-v2'
if ((Get-Content -LiteralPath (Join-Path $bridgeOut 'queue.json') -Raw | ConvertFrom-Json).status -ne 'prepared_not_started') { throw 'Preserve existing experiment' }
if ((Get-FileHash -LiteralPath (Join-Path $bridgeOut 'protocol.json')).Hash -ne (Get-Content -LiteralPath (Join-Path $bridgeOut 'protocol.sha256') -Raw).Trim()) { throw 'Protocol changed' }
if (Get-ScheduledTask -TaskName $bridgeTask -ErrorAction SilentlyContinue) { throw 'Task exists' }
$bridgeArgs = '-u "' + (Join-Path $bridgeRoot 'data\scripts\run_detector_classifier_bridge_v2.py') + '" scheduled-run'
$bridgeAction = New-ScheduledTaskAction -Execute 'D:\Anaconda3\python.exe' -Argument $bridgeArgs -WorkingDirectory $bridgeRoot
$bridgeSettings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 6) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$bridgePrincipal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $bridgeTask -Action $bridgeAction -Settings $bridgeSettings -Principal $bridgePrincipal -Description 'One-shot native validation source restore and fixed cached detector-to-classifier bridge; no recurrence' | Select-Object TaskName,State
Start-ScheduledTask -TaskName $bridgeTask
Get-ScheduledTask -TaskName $bridgeTask | Select-Object TaskName,State
