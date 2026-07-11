@echo off
chcp 65001 >nul
echo.
echo 300M magyar LM - Teszt CLI
echo ============================
echo.
ssh -i "%USERPROFILE%\.ssh\neura_remote_key" -t neura@192.168.0.142 "C:\Users\neura\Python311\python.exe -u C:\NeuraNode\hemna_bench\test_lm_cli.py"
pause
