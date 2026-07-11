"""sin(x) regression: MLP vs BSpline - ahol a standard MLP gyenge"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, math, time
sys.path.insert(0, '.')
from hemna_v3 import VectorizedBSplineLayer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Adat: y = sin(x) a [-10, 10] tartomanyon
torch.manual_seed(42)
X_train = torch.rand(2000, 1) * 20 - 10  # [-10, 10]
y_train = torch.sin(X_train)
X_test = torch.linspace(-10, 10, 500).reshape(-1, 1)
y_test = torch.sin(X_test)

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

def count_params(*modules):
    return sum(p.numel() for m in modules for p in m.parameters())

epochs = 2000

print(f"\n{'Modell':<30} {'Params':<12} {'Layer':<15} {'Test Loss':<15}")
print("-" * 75)

results = []

# === 1. MLP (ReLU) - 1 layer ===
for width in [4, 8, 16, 32]:
    mlp = nn.Sequential(
        nn.Linear(1, width), nn.ReLU(),
        nn.Linear(width, 1)
    ).to(device)
    p = count_params(mlp)
    opt = torch.optim.Adam(mlp.parameters(), lr=0.01)
    
    for ep in range(epochs):
        opt.zero_grad()
        loss = F.mse_loss(mlp(X_train), y_train)
        loss.backward()
        opt.step()
    
    with torch.no_grad():
        test_loss = F.mse_loss(mlp(X_test), y_test).item()
    print(f"{'MLP ReLU 1 layer':<30} {p:<12} {f'width={width}':<15} {test_loss:<15.6f}")
    results.append(('MLP', width, p, test_loss))

# === 2. MLP (ReLU) - 2 layers ===
for width in [8, 16, 32]:
    mlp = nn.Sequential(
        nn.Linear(1, width), nn.ReLU(),
        nn.Linear(width, width), nn.ReLU(),
        nn.Linear(width, 1)
    ).to(device)
    p = count_params(mlp)
    opt = torch.optim.Adam(mlp.parameters(), lr=0.01)
    
    for ep in range(epochs):
        opt.zero_grad()
        loss = F.mse_loss(mlp(X_train), y_train)
        loss.backward()
        opt.step()
    
    with torch.no_grad():
        test_loss = F.mse_loss(mlp(X_test), y_test).item()
    print(f"{'MLP ReLU 2 layers':<30} {p:<12} {f'width={width}':<15} {test_loss:<15.6f}")
    results.append(('MLP2', width, p, test_loss))

# === 3. BSpline only (T3) - VectorizedBSplineLayer ===
for grid_size in [4, 8, 16, 32]:
    spline = VectorizedBSplineLayer(1, 1, grid_min=-10, grid_max=10, 
                                     grid_size=grid_size, degree=3).to(device)
    p = count_params(spline)
    opt = torch.optim.Adam(spline.parameters(), lr=0.01)
    
    for ep in range(epochs):
        opt.zero_grad()
        loss = F.mse_loss(spline(X_train), y_train)
        loss.backward()
        opt.step()
    
    with torch.no_grad():
        test_loss = F.mse_loss(spline(X_test), y_test).item()
    print(f"{'BSpline (T3)':<30} {p:<12} {f'grid={grid_size}':<15} {test_loss:<15.6f}")
    results.append(('BSpline', grid_size, p, test_loss))

# === 4. BSpline degree=5 (magasabb fokszam) ===
for grid_size in [8, 16]:
    spline = VectorizedBSplineLayer(1, 1, grid_min=-10, grid_max=10,
                                     grid_size=grid_size, degree=5).to(device)
    p = count_params(spline)
    opt = torch.optim.Adam(spline.parameters(), lr=0.01)
    
    for ep in range(epochs):
        opt.zero_grad()
        loss = F.mse_loss(spline(X_train), y_train)
        loss.backward()
        opt.step()
    
    with torch.no_grad():
        test_loss = F.mse_loss(spline(X_test), y_test).item()
    print(f"{'BSpline degree=5':<30} {p:<12} {f'grid={grid_size}':<15} {test_loss:<15.6f}")
    results.append(('BSpline5', grid_size, p, test_loss))

print(f"\n=== OSSZEGZES ===")
# Legjobb MLP
best_mlp = min([r for r in results if r[0] == 'MLP'], key=lambda x: x[3])
best_mlp2 = min([r for r in results if r[0] == 'MLP2'], key=lambda x: x[3])
best_bs = min([r for r in results if r[0] in ('BSpline', 'BSpline5')], key=lambda x: x[3])
print(f"  Legjobb MLP 1 layer:  {best_mlp[2]} param, loss={best_mlp[3]:.6f} (width={best_mlp[1]})")
print(f"  Legjobb MLP 2 layer:  {best_mlp2[2]} param, loss={best_mlp2[3]:.6f} (width={best_mlp2[1]})")
print(f"  Legjobb BSpline:      {best_bs[2]} param, loss={best_bs[3]:.6f} (grid={best_bs[1]})")
print()
if best_bs[3] < best_mlp[3] and best_bs[3] < best_mlp2[3]:
    print(">>> A BSpline VERI az MLP-t ezen a feladaton!")
else:
    print(">>> Az MLP nyert. (rossz irany)")
