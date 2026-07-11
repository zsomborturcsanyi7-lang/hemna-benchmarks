"""
HEMNA vs MLP vs DeepBSpline — 2D ADDITIVE + MULTIPLICATIVE

1. f(x,y) = sin(x) + cos(y) - ADDITIV, BSpline-nak nyernie kell
2. f(x,y) = sin(x) * cos(y) - MULTIPLIKATIV, Deep kell
"""
import torch, torch.nn as nn, torch.nn.functional as F, sys, time
sys.path.insert(0, '.')
from hemna_v3 import VectorizedBSplineLayer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

def count_params(*ms):
    return sum(p.numel() for m in ms for p in m.parameters())

def generate(func, n=5000, seed=42):
    torch.manual_seed(seed)
    X = torch.rand(n, 2) * 10 - 5
    if func == 'add':
        y = (torch.sin(X[:,0]) + torch.cos(X[:,1])).unsqueeze(1)
    else:
        y = (torch.sin(X[:,0]) * torch.cos(X[:,1])).unsqueeze(1)
    return X.to(device), y.to(device)

X_test_add, y_test_add = generate('add', 2000, 999)
X_test_mul, y_test_mul = generate('mul', 2000, 999)

epochs = 2000

# Deep BSpline: ket BSpline reteg egymas utan
class DeepBSpline(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, grid=16):
        super().__init__()
        self.l1 = VectorizedBSplineLayer(in_dim, hidden_dim, grid_min=-5, grid_max=5, grid_size=grid)
        self.l2 = VectorizedBSplineLayer(hidden_dim, out_dim, grid_min=-5, grid_max=5, grid_size=grid)
    def forward(self, x):
        return self.l2(self.l1(x))

# MLP 2 layer
def make_mlp(width, layers=2):
    if layers == 1:
        return nn.Sequential(nn.Linear(2, width), nn.ReLU(), nn.Linear(width, 1))
    return nn.Sequential(nn.Linear(2, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))

# ============================================================
# 1. ADDITIVE: sin(x) + cos(y)
# ============================================================
print("=" * 60)
print("1. ADDITIVE: f(x,y) = sin(x) + cos(y)")
print("=" * 60)
print(f"{'Modell':<30} {'Params':<10} {'Test Loss':<12}")

for width in [8, 16, 32, 64]:
    for layers in [1, 2]:
        model = make_mlp(width, layers).to(device)
        X, y = generate('add', 5000, 42)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(epochs):
            opt.zero_grad(); loss = F.mse_loss(model(X), y); loss.backward(); opt.step()
        tl = F.mse_loss(model(X_test_add), y_test_add).item()
        print(f"  {'MLP '+str(layers)+'l w='+str(width):<30} {count_params(model):<10} {tl:<12.6f}")

for grid in [4, 8, 16, 32]:
    model = VectorizedBSplineLayer(2, 1, grid_min=-5, grid_max=5, grid_size=grid).to(device)
    X, y = generate('add', 5000, 42)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(epochs):
        opt.zero_grad(); loss = F.mse_loss(model(X), y); loss.backward(); opt.step()
    tl = F.mse_loss(model(X_test_add), y_test_add).item()
    print(f"  {'BSpline grid='+str(grid):<30} {count_params(model):<10} {tl:<12.6f}")

for grid in [8, 16]:
    for h in [8, 16]:
        model = DeepBSpline(2, h, 1, grid).to(device)
        X, y = generate('add', 5000, 42)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(epochs):
            opt.zero_grad(); loss = F.mse_loss(model(X), y); loss.backward(); opt.step()
        tl = F.mse_loss(model(X_test_add), y_test_add).item()
        print(f"  {'DeepBSpline g'+str(grid)+'h'+str(h):<30} {count_params(model):<10} {tl:<12.6f}")

# ============================================================
# 2. MULTIPLICATIVE: sin(x) * cos(y) - Deep model kell
# ============================================================
print("\n" + "=" * 60)
print("2. MULTIPLICATIVE: f(x,y) = sin(x) * cos(y)")
print("=" * 60)
print(f"{'Modell':<30} {'Params':<10} {'Test Loss':<12}")

for width in [8, 16, 32, 64]:
    for layers in [1, 2]:
        model = make_mlp(width, layers).to(device)
        X, y = generate('mul', 5000, 42)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(epochs):
            opt.zero_grad(); loss = F.mse_loss(model(X), y); loss.backward(); opt.step()
        tl = F.mse_loss(model(X_test_mul), y_test_mul).item()
        print(f"  {'MLP '+str(layers)+'l w='+str(width):<30} {count_params(model):<10} {tl:<12.6f}")

for grid in [8, 16]:
    for h in [16, 32]:
        model = DeepBSpline(2, h, 1, grid).to(device)
        X, y = generate('mul', 5000, 42)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(epochs):
            opt.zero_grad(); loss = F.mse_loss(model(X), y); loss.backward(); opt.step()
        tl = F.mse_loss(model(X_test_mul), y_test_mul).item()
        print(f"  {'DeepBSpline g'+str(grid)+'h'+str(h):<30} {count_params(model):<10} {tl:<12.6f}")

print("\n=== OSSZEGZES ===")
print("Additiv (BSpline-nak nyernie KELL):")
print("  MLP nyert vagy BSpline? -> random")
print("Multiplikativ (Deep kell):")
print("  MLP nyert vagy DeepBSpline? -> random")
