@echo off
REM GPU warmup
C:\Users\neura\Python311\python.exe -c "import torch; a=torch.randn(10000,10000,device='cuda'); [torch.mm(a,a) for _ in range(30)]; torch.cuda.synchronize()"
REM Clear old log
if exist C:\NeuraNode\hemna_bench\continue_300m_log.txt del C:\NeuraNode\hemna_bench\continue_300m_log.txt
REM Run training
C:\Users\neura\Python311\python.exe -u C:\NeuraNode\hemna_bench\continue_300m.py
