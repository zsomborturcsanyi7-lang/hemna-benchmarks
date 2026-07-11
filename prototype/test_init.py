import time, sys
sys.path.insert(0, '.')
from hemna_v3 import HEMNAv3

print("Creating model...")
s = time.time()
m = HEMNAv3(784, [32], 10, 0.001, 0.001, 0.002, 100)
print(f"Created: {time.time()-s:.1f}s")
print(f"Params: {sum(p.numel() for p in m.parameters())}")
