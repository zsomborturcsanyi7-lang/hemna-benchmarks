Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "C:\NeuraNode\hemna_bench"
WshShell.Run "C:\Users\neura\Python311\python.exe -u C:\NeuraNode\hemna_bench\continue_300m.py", 0, False
