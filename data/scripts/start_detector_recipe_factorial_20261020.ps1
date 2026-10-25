$ErrorActionPreference = 'Stop'
$recipeRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$recipeOut = Join-Path $recipeRoot 'runs\experiments\detector_recipe_factorial_20261020_v1'
$recipeTask = 'MarsKG-detector-recipe-factorial-20261020-v1'
if ((Get-Content -LiteralPath (Join-Path $recipeOut 'queue.json') -Raw | ConvertFrom-Json).status -ne 'prepared_not_started') { throw 'Preserve existing run' }
if (Get-ScheduledTask -TaskName $recipeTask -ErrorAction SilentlyContinue) { throw 'Task already exists' }
$recipeAction = New-ScheduledTaskAction -Execute 'D:\Anaconda3\python.exe' -Argument ('-X utf8 -u "' + (Join-Path $PSScriptRoot 'run_detector_recipe_factorial_20261020.py') + '" scheduled-run') -WorkingDirectory $recipeRoot
$recipeSettings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 36) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$recipePrincipal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $recipeTask -Action $recipeAction -Settings $recipeSettings -Principal $recipePrincipal -Description 'One finite four-arm augmentation/update-batch study; real short tests, epoch recovery and common validation; no periodic trigger' | Out-Null
Start-ScheduledTask -TaskName $recipeTask
Get-ScheduledTask -TaskName $recipeTask | Select-Object TaskName,State
