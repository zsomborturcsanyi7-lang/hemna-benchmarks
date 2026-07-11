"""Vektorizált BSpline sebesség teszt"""
import torch, time, sys
sys.path.insert(0, '.')
from hemna_v3 import VectorizedBSplineLayer

# Régi Tier3Linear (ha még elérhető)
try:
    from hemna_v3_old import Tier3Linear as OldTier3Linear
    has_old = True
except:
    has_old = False

print("=== Vektorizált BSpline sebesség teszt ===")

# === Small: 100 dim, 8 out ===
t3 = VectorizedBSplineLayer(100, 8).cuda()
x = torch.randn(128, 100).cuda()

for _ in range(10): t3(x)
torch.cuda.synchronize()
s = time.time()
for _ in range(100): t3(x)
torch.cuda.synchronize()
ms = (time.time() - s) / 100 * 1000
print(f"  (100in, 8out, batch=128): {ms:.3f} ms")
print(f"  Parameters: {sum(p.numel() for p in t3.parameters())}")

# === Large: 784 dim, 16 out (MNIST méret) ===
t3b = VectorizedBSplineLayer(784, 16).cuda()
xb = torch.randn(128, 784).cuda()

for _ in range(10): t3b(xb)
torch.cuda.synchronize()
s = time.time()
for _ in range(100): t3b(xb)
torch.cuda.synchronize()
ms = (time.time() - s) / 100 * 1000
print(f"\n  (784in, 16out, batch=128): {ms:.3f} ms")
print(f"  Parameters: {sum(p.numel() for p in t3b.parameters())}")

# === Hatalmas: 784 dim, 64 out (teljes réteg) ===
t3c = VectorizedBSplineLayer(784, 64).cuda()
xc = torch.randn(128, 784).cuda()

for _ in range(10): t3c(xc)
torch.cuda.synchronize()
s = time.time()
for _ in range(100): t3c(xc)
torch.cuda.synchronize()
ms = (time.time() - s) / 100 * 1000
print(f"\n  (784in, 64out, batch=128): {ms:.3f} ms")
print(f"  Parameters: {sum(p.numel() for p in t3c.parameters())}")
