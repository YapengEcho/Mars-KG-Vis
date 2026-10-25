$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$out = Join-Path $root 'runs\experiments\classifier_encoder_arms_20261015_v1'
$task = 'MarsKG-classifier-encoder-arms-20261015-v1'
if ((Get-Content -LiteralPath (Join-Path $out 'queue.json') -Raw | ConvertFrom-Json).status -ne 'prepared_not_started') { throw 'Preserve existing run' }
if ((Get-Content -LiteralPath (Join-Path $out 'readiness.json') -Raw | ConvertFrom-Json).status -ne 'passed_real_GPU_smoke') { throw 'GPU smoke missing' }
if ((Get-FileHash -LiteralPath (Join-Path $out 'protocol.json')).Hash -ne (Get-Content -LiteralPath (Join-Path $out 'protocol.sha256') -Raw).Trim()) { throw 'Protocol changed' }
if (Get-ScheduledTask -TaskName $task -ErrorAction SilentlyContinue) { throw 'Task exists' }
$arguments = '-u "' + (Join-Path $root 'data\scripts\run_classifier_encoder_arms.py') + '" scheduled-run'
$action = New-ScheduledTaskAction -Execute 'D:\Anaconda3\python.exe' -Argument $arguments -WorkingDirectory $root
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 12) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $task -Action $action -Settings $settings -Principal $principal -Description 'One-shot encoder-parallel multiseed head/LoRA/layer4 four-class comparison; no recurring trigger' | Select-Object TaskName,State
Start-ScheduledTask -TaskName $task
Get-ScheduledTask -TaskName $task | Select-Object TaskName,State
