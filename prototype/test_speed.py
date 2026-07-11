import time, torch, sys
sys.path.insert(0, '.')
from hemna_v3 import HEMNAv3

device = 'cuda'
m = HEMNAv3(784, [32], 10, 0.001, 0.001, 0.002, 100).to(device)
x = torch.randn(128, 784).to(device)

# Warmup
for _ in range(5):
    m(x)
torch.cuda.synchronize()

# Measure
s = time.time()
for _ in range(10):
    m(x)
torch.cuda.synchronize()
ms = (time.time() - s) / 10 * 1000
print(f"Forward: {ms:.1f}ms per batch (128)")
