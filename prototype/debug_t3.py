"""Debug: Tier3Linear creation time"""
import sys, time
sys.path.insert(0, '.')
from hemna_v3 import Tier3Linear

print("Creating Tier3Linear(100, 8)...")
s = time.time()
t3 = Tier3Linear(100, 8)
print(f"Created in {time.time()-s:.1f}s")
print(f"Params: {sum(p.numel() for p in t3.parameters())}")

print("\nForward test...")
import torch
x = torch.randn(128, 100).cuda()
t3 = t3.cuda()
s = time.time()
out = t3(x)
print(f"Forward: {out.shape} in {(time.time()-s)*1000:.1f}ms")
