"""Wake GPU from 210 MHz to full speed"""
import torch
t = torch.tensor([5000*5000], device='cuda')
for _ in range(30):
    torch.mm(torch.randn(5000,5000,device='cuda'), torch.randn(5000,5000,device='cuda')).sum().item()
import os
os.system('powershell -Command "nvidia-smi --query-gpu=clocks.current.graphics --format=csv,noheader"')
print("GPU FELÉBRESZTVE ✅")
