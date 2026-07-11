"""
HEMNA v3 benchmark — T1+T2+T3 vs Standard MLP
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, time
sys.path.insert(0, '.')
from hemna_v3 import Tier1Linear, VectorizedBSplineLayer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Szintetikus adat: 2000 minta, 100 dimenzió, 10 osztály
torch.manual_seed(42)
X = torch.randn(2000, 100).to(device)
y = torch.randint(0, 10, (2000,)).to(device)

# ===== 1. Standard MLP =====
print("\n=== 1. Standard MLP (1 réteg, 64 ReLU) ===")
mlp = nn.Sequential(
    nn.Linear(100, 64), nn.ReLU(),
    nn.Linear(64, 10)
).to(device)
opt = torch.optim.Adam(mlp.parameters(), lr=0.001)
mlp_params = sum(p.numel() for p in mlp.parameters())
print(f"  Paraméterek: {mlp_params}")

start = time.time()
for step in range(2000):
    pred = mlp(X)
    loss = F.cross_entropy(pred, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 500 == 0:
        acc = (pred.argmax(1) == y).float().mean().item()
        print(f"  Step {step:4d}: loss={loss.item():.4f} acc={acc:.2%}")
print(f"  Idő: {time.time()-start:.1f}s")
with torch.no_grad():
    acc = (mlp(X).argmax(1) == y).float().mean().item()
print(f"  Final: {acc:.2%}")

# ===== 2. HEMNA T1+T2 (fix tier) =====
print("\n=== 2. HEMNA (32 T1 + 32 T2) ===")
torch.manual_seed(42)
t1 = Tier1Linear(100, 32).to(device)
t2 = nn.Linear(100, 32).to(device)
output = nn.Linear(64, 10).to(device)
opt = torch.optim.Adam(
    list(t1.parameters()) + list(t2.parameters()) + list(output.parameters()), lr=0.001)

start = time.time()
for step in range(2000):
    h = torch.cat([t1(X), F.relu(t2(X))], dim=-1)
    pred = output(h)
    loss = F.cross_entropy(pred, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 500 == 0:
        acc = (pred.argmax(1) == y).float().mean().item()
        print(f"  Step {step:4d}: loss={loss.item():.4f} acc={acc:.2%}")
print(f"  Idő: {time.time()-start:.1f}s")
with torch.no_grad():
    acc = (output(torch.cat([t1(X), F.relu(t2(X))], -1)).argmax(1) == y).float().mean().item()
print(f"  Final: {acc:.2%}")

# ===== 3. HEMNA T1+T2+T3 (vektorizált BSpline) =====
print("\n=== 3. HEMNA (16 T1 + 16 T2 + 16 T3) ===")
torch.manual_seed(42)
t1c = Tier1Linear(100, 16).to(device)
t2c = nn.Linear(100, 16).to(device)
t3c = VectorizedBSplineLayer(100, 16).to(device)
outputc = nn.Linear(48, 10).to(device)
optc = torch.optim.Adam(
    list(t1c.parameters()) + list(t2c.parameters()) + 
    list(t3c.parameters()) + list(outputc.parameters()), lr=0.001)
h3_params = sum(p.numel() for p in 
    list(t1c.parameters()) + list(t2c.parameters()) + 
    list(t3c.parameters()) + list(outputc.parameters()))
print(f"  Paraméterek: {h3_params} (MLP: {mlp_params})")

start = time.time()
for step in range(2000):
    h = torch.cat([t1c(X), F.relu(t2c(X)), t3c(X)], dim=-1)
    pred = outputc(h)
    loss = F.cross_entropy(pred, y)
    optc.zero_grad()
    loss.backward()
    optc.step()
    if step % 500 == 0:
        acc = (pred.argmax(1) == y).float().mean().item()
        print(f"  Step {step:4d}: loss={loss.item():.4f} acc={acc:.2%}")
print(f"  Idő: {time.time()-start:.1f}s")
with torch.no_grad():
    acc = (outputc(torch.cat([t1c(X), F.relu(t2c(X)), t3c(X)], -1)).argmax(1) == y).float().mean().item()
print(f"  Final: {acc:.2%}")

print("\n=== ÖSSZEGZÉS ===")
with torch.no_grad():
    acc1 = (mlp(X).argmax(1) == y).float().mean().item()
    acc2 = (output(torch.cat([t1(X), F.relu(t2(X))], -1)).argmax(1) == y).float().mean().item()
    acc3 = (outputc(torch.cat([t1c(X), F.relu(t2c(X)), t3c(X)], -1)).argmax(1) == y).float().mean().item()
print(f"  1. Standard MLP ({mlp_params} param):         {acc1:.2%}")
print(f"  2. HEMNA T1+T2 (32+32):                       {acc2:.2%}")
print(f"  3. HEMNA T1+T2+T3 (16+16+16): {h3_params} param: {acc3:.2%}")
