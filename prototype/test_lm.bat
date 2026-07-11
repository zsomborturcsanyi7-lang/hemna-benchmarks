@echo off
REM 300M magyar LM teszt CLI
REM Hasznalat: test_lm.bat
echo.
echo 300M magyar LM - teszt CLI
echo ============================
echo.
echo Betoltes es teszt inditasa a NEURA gepen...
echo.
ssh -i %USERPROFILE%\.ssh\neura_remote_key neura@192.168.0.142 "C:\Users\neura\Python311\python.exe -u C:\NeuraNode\hemna_bench\test_lm_cli.py"
