$action = New-ScheduledTaskAction -Execute 'C:\Users\neura\Python311\python.exe' -Argument '-u C:\NeuraNode\hemna_bench\continue_300m.py'
$trigger = New-ScheduledTaskTrigger -Once -At ((Get-Date).AddMinutes(1))
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
Unregister-ScheduledTask -TaskName 'NEURA_TRAIN_V3' -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName 'NEURA_TRAIN_V3' -Action $action -Trigger $trigger -Settings $settings -User 'neura' -RunLevel Limited
Write-Output 'TASK CREATED'
