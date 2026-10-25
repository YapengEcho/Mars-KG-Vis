$ErrorActionPreference = 'Stop'
$headRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$headOut = Join-Path $headRoot 'runs\experiments\classifier_head_rebalance_native40_20261015_v1'
$headTask = 'MarsKG-classifier-head-rebalance-native40-20261015-v1'
if ((Get-Content -LiteralPath (Join-Path $headOut 'queue.json') -Raw | ConvertFrom-Json).status -ne 'prepared_not_started') { throw 'Preserve existing run' }
$headReady = Get-Content -LiteralPath (Join-Path $headOut 'readiness.json') -Raw | ConvertFrom-Json
$headHash = (Get-Content -LiteralPath (Join-Path $headOut 'protocol.sha256') -Raw).Trim()
if ($headReady.status -ne 'passed_real_GPU_smoke' -or $headReady.protocol_sha256 -ne $headHash) { throw 'GPU smoke missing or stale' }
if ((Get-FileHash -LiteralPath (Join-Path $headOut 'protocol.json')).Hash -ne $headHash) { throw 'Protocol changed' }
if (Get-ScheduledTask -TaskName $headTask -ErrorAction SilentlyContinue) { throw 'Task exists' }
$headArguments = '-u "' + (Join-Path $headRoot 'data\scripts\run_classifier_head_rebalance_native40.py') + '" scheduled-run'
$headAction = New-ScheduledTaskAction -Execute 'D:\Anaconda3\python.exe' -Argument $headArguments -WorkingDirectory $headRoot
$headSettings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 12) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$headPrincipal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $headTask -Action $headAction -Settings $headSettings -Principal $headPrincipal -Description 'One-shot fixed-LoRA head class-weight comparison, expert labels retained, no recurring trigger' | Select-Object TaskName,State
Start-ScheduledTask -TaskName $headTask
Get-ScheduledTask -TaskName $headTask | Select-Object TaskName,State
