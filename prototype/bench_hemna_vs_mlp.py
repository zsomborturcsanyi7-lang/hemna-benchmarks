"""
HEMNA v3 benchmark — T1+T2 vs Standard MLP (T3 nélkül)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, time
sys.path.insert(0, '.')
from hemna_v3 import Tier1Linear

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
print(f"  Final accuracy: {acc:.2%}")

# ===== 2. HEMNA T1+T2 (fix tier) =====
print("\n=== 2. HEMNA (32 T1 + 32 T2, fix) ===")
torch.manual_seed(42)
t1 = Tier1Linear(100, 32).to(device)
t2 = nn.Linear(100, 32).to(device)
output = nn.Linear(64, 10).to(device)
params = list(t1.parameters()) + list(t2.parameters()) + list(output.parameters())
opt = torch.optim.Adam(params, lr=0.001)
h_params = sum(p.numel() for p in params)
print(f"  Paraméterek: {h_params}")

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
    acc = (output(torch.cat([t1(X), F.relu(t2(X))], dim=-1)).argmax(1) == y).float().mean().item()
print(f"  Final accuracy: {acc:.2%}")

# ===== 3. HEMNA T1+T2 ugyanannyi paraméterrel =====
print("\n=== 3. HEMNA (46 T1 + 46 T2) = ugyanannyi paraméter mint MLP ===")
# MLP: 100*64 + 64 + 64*10 + 10 = 6400+64+640+10 = 7114
# Képlet: (100*n_t1) + n_t1 + (100*n_t2) + n_t2 + (n_t1+n_t2)*10 + 10
# n_t1 + n_t2 = 64, n_t1=n_t2=32 → 100*32+32+100*32+32+64*10+10 = 3200+32+3200+32+640+10 = 7114 ✓
torch.manual_seed(42)
t1b = Tier1Linear(100, 46).to(device)
t2b = nn.Linear(100, 46).to(device)
outputb = nn.Linear(92, 10).to(device)
paramsb = list(t1b.parameters()) + list(t2b.parameters()) + list(outputb.parameters())
optb = torch.optim.Adam(paramsb, lr=0.001)
h2_params = sum(p.numel() for p in paramsb)
print(f"  Paraméterek: {h2_params} (MLP: {mlp_params})")

start = time.time()
for step in range(2000):
    h = torch.cat([t1b(X), F.relu(t2b(X))], dim=-1)
    pred = outputb(h)
    loss = F.cross_entropy(pred, y)
    optb.zero_grad()
    loss.backward()
    optb.step()
    if step % 500 == 0:
        acc = (pred.argmax(1) == y).float().mean().item()
        print(f"  Step {step:4d}: loss={loss.item():.4f} acc={acc:.2%}")
print(f"  Idő: {time.time()-start:.1f}s")
with torch.no_grad():
    acc = (outputb(torch.cat([t1b(X), F.relu(t2b(X))], dim=-1)).argmax(1) == y).float().mean().item()
print(f"  Final accuracy: {acc:.2%}")

print("\n=== ÖSSZEGZÉS ===")
print(f"  1. Standard MLP:     {mlp_params} param — acc: ", end="")
with torch.no_grad(): print(f"{ (mlp(X).argmax(1)==y).float().mean().item():.2%}")
print(f"  2. HEMNA T1+T2:      {h_params} param — acc: ", end="")
with torch.no_grad(): print(f"{ (output(torch.cat([t1(X),F.relu(t2(X))],-1)).argmax(1)==y).float().mean().item():.2%}")
print(f"  3. HEMNA (equal):    {h2_params} param — acc: ", end="")
with torch.no_grad(): print(f"{ (outputb(torch.cat([t1b(X),F.relu(t2b(X))],-1)).argmax(1)==y).float().mean().item():.2%}")
