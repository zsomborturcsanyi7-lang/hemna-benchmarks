$taskName = 'NEURA_TRAIN_V4'
$action = New-ScheduledTaskAction -Execute 'C:\Users\neura\Python311\python.exe' -Argument '-u C:\NeuraNode\hemna_bench\continue_300m.py' -WorkingDirectory 'C:\NeuraNode\hemna_bench'
$trigger = New-ScheduledTaskTrigger -Once -At ((Get-Date).AddMinutes(1))
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -Hidden
$principal = New-ScheduledTaskPrincipal -UserId 'neura' -RunLevel Limited -LogonType Interactive
Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal
Write-Output "TASK_CREATED: $taskName"
