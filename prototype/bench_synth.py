"""
HEMNA v3 — Gyors benchmark MNIST-en (csak 1 epoch, kevesebb neuron)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, time
sys.path.insert(0, '.')
from hemna_v3 import Tier1Linear, Tier2Linear, Tier3Linear

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Egyszerű szintetikus adat (nem MNIST, gyors)
# 1000 minta, 100 dimenzió, 10 osztály
torch.manual_seed(42)
X = torch.randn(1000, 100)
y = torch.randint(0, 10, (1000,))

# ===== Standard MLP =====
print("\n=== Standard MLP ===")
mlp = nn.Sequential(
    nn.Linear(100, 32), nn.ReLU(),
    nn.Linear(32, 10)
).to(device)
opt = torch.optim.Adam(mlp.parameters(), lr=0.001)

start = time.time()
for step in range(500):
    pred = mlp(X.to(device))
    loss = F.cross_entropy(pred, y.to(device))
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 100 == 0:
        acc = (pred.argmax(1) == y.to(device)).float().mean().item()
        print(f"  Step {step}: loss={loss.item():.4f} acc={acc:.2%}")
print(f"MLP time: {time.time()-start:.1f}s")

# ===== HEMNA v3 fixed =====
print("\n=== HEMNA v3 (8 T1 + 8 T2 + 8 T3) ===")
torch.manual_seed(42)
t1 = Tier1Linear(100, 8).to(device)
t2 = Tier2Linear(100, 8).to(device)
t3 = Tier3Linear(100, 8).to(device)  # 8×100 BSpline = 800 BSpline!
output = nn.Linear(24, 10).to(device)
params = list(t1.parameters()) + list(t2.parameters()) + list(t3.parameters()) + list(output.parameters())
opt = torch.optim.Adam(params, lr=0.001)
n_params = sum(p.numel() for p in params)
print(f"  Parameters: {n_params}")

start = time.time()
for step in range(500):
    h = torch.cat([t1(X.to(device)), t2(X.to(device)), t3(X.to(device))], dim=-1)
    pred = output(h)
    loss = F.cross_entropy(pred, y.to(device))
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 100 == 0:
        acc = (pred.argmax(1) == y.to(device)).float().mean().item()
        print(f"  Step {step}: loss={loss.item():.4f} acc={acc:.2%}")
print(f"HEMNA time: {time.time()-start:.1f}s")

# ===== HEMNA only T1+T2 (no T3) =====
print("\n=== HEMNA (16 T1 + 16 T2, no T3) ===")
torch.manual_seed(42)
t1b = Tier1Linear(100, 16).to(device)
t2b = Tier2Linear(100, 16).to(device)
output2 = nn.Linear(32, 10).to(device)
params2 = list(t1b.parameters()) + list(t2b.parameters()) + list(output2.parameters())
opt2 = torch.optim.Adam(params2, lr=0.001)

start = time.time()
for step in range(500):
    h = torch.cat([t1b(X.to(device)), t2b(X.to(device))], dim=-1)
    pred = output2(h)
    loss = F.cross_entropy(pred, y.to(device))
    opt2.zero_grad()
    loss.backward()
    opt2.step()
    if step % 100 == 0:
        acc = (pred.argmax(1) == y.to(device)).float().mean().item()
        print(f"  Step {step}: loss={loss.item():.4f} acc={acc:.2%}")
print(f"HEMNA (T1+T2) time: {time.time()-start:.1f}s")
