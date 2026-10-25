$ErrorActionPreference = 'Stop'
$nativeRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$nativeOut = Join-Path $nativeRoot 'runs\experiments\classifier_lora_native40_multiseed_20261015_v1'
$nativeTask = 'MarsKG-classifier-lora-native40-20261015-v1'
if ((Get-Content -LiteralPath (Join-Path $nativeOut 'queue.json') -Raw | ConvertFrom-Json).status -ne 'prepared_not_started') { throw 'Preserve existing run' }
$nativeReady = Get-Content -LiteralPath (Join-Path $nativeOut 'readiness.json') -Raw | ConvertFrom-Json
if ($nativeReady.status -ne 'passed_real_GPU_smoke') { throw 'GPU smoke missing' }
$nativeHash = (Get-Content -LiteralPath (Join-Path $nativeOut 'protocol.sha256') -Raw).Trim()
if ((Get-FileHash -LiteralPath (Join-Path $nativeOut 'protocol.json')).Hash -ne $nativeHash -or $nativeReady.protocol_sha256 -ne $nativeHash) { throw 'Protocol changed' }
if (Get-ScheduledTask -TaskName $nativeTask -ErrorAction SilentlyContinue) { throw 'Task exists' }
$nativeArguments = '-u "' + (Join-Path $nativeRoot 'data\scripts\run_classifier_lora_native40.py') + '" scheduled-run'
$nativeAction = New-ScheduledTaskAction -Execute 'D:\Anaconda3\python.exe' -Argument $nativeArguments -WorkingDirectory $nativeRoot
$nativeSettings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 12) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$nativePrincipal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $nativeTask -Action $nativeAction -Settings $nativeSettings -Principal $nativePrincipal -Description 'One-shot native40 spatial classification comparison; reuse seed20261015, train two new seeds; no recurring trigger' | Select-Object TaskName,State
Start-ScheduledTask -TaskName $nativeTask
Get-ScheduledTask -TaskName $nativeTask | Select-Object TaskName,State
