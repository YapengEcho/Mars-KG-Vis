$ErrorActionPreference = 'Stop'
$loraRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$loraOut = Join-Path $loraRoot 'runs\experiments\classifier_lora_20261015_v1'
$loraTask = 'MarsKG-classifier-lora-20261015-v1'
if ((Get-Content -LiteralPath (Join-Path $loraOut 'queue.json') -Raw | ConvertFrom-Json).status -ne 'prepared_not_started') { throw 'Preserve existing run' }
if ((Get-Content -LiteralPath (Join-Path $loraOut 'readiness.json') -Raw | ConvertFrom-Json).status -ne 'passed_real_GPU_smoke') { throw 'GPU smoke missing' }
if ((Get-FileHash -LiteralPath (Join-Path $loraOut 'protocol.json')).Hash -ne (Get-Content -LiteralPath (Join-Path $loraOut 'protocol.sha256') -Raw).Trim()) { throw 'Protocol changed' }
if (Get-ScheduledTask -TaskName $loraTask -ErrorAction SilentlyContinue) { throw 'Task exists' }
$loraArguments = '-u "' + (Join-Path $loraRoot 'data\scripts\run_classifier_lora.py') + '" scheduled-run'
$loraAction = New-ScheduledTaskAction -Execute 'D:\Anaconda3\python.exe' -Argument $loraArguments -WorkingDirectory $loraRoot
$loraSettings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 12) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$loraPrincipal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $loraTask -Action $loraAction -Settings $loraSettings -Principal $loraPrincipal -Description 'One-shot nested spatial head-only and LoRA training; no recurring trigger' | Select-Object TaskName,State
Start-ScheduledTask -TaskName $loraTask
Get-ScheduledTask -TaskName $loraTask | Select-Object TaskName,State
