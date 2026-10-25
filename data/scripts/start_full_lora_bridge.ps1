$ErrorActionPreference = 'Stop'
$bridgeRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$bridgeOut = Join-Path $bridgeRoot 'runs\experiments\full_lora_bridge_20261015_v1'
$bridgeTask = 'MarsKG-full-lora-bridge-20261015-v1'
if ((Get-Content -LiteralPath (Join-Path $bridgeOut 'queue.json') -Raw | ConvertFrom-Json).status -ne 'prepared_not_started') { throw 'Preserve existing run' }
if ((Get-Content -LiteralPath (Join-Path $bridgeOut 'readiness.json') -Raw | ConvertFrom-Json).status -ne 'passed_real_GPU_smoke') { throw 'GPU smoke missing' }
if (Get-ScheduledTask -TaskName $bridgeTask -ErrorAction SilentlyContinue) { throw 'Task exists' }
$bridgeAction = New-ScheduledTaskAction -Execute 'D:\Anaconda3\python.exe' -Argument ('-u "' + (Join-Path $bridgeRoot 'data\scripts\run_full_lora_bridge.py') + '" scheduled-run') -WorkingDirectory $bridgeRoot
$bridgeSettings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 12) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$bridgePrincipal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $bridgeTask -Action $bridgeAction -Settings $bridgeSettings -Principal $bridgePrincipal -Description 'One-shot full-training LoRA ROI bridge, predeclared epochs, no recurring trigger' | Select-Object TaskName,State
Start-ScheduledTask -TaskName $bridgeTask
