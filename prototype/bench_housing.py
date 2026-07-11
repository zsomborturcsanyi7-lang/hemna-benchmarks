"""
Valos benchmark: California Housing regression
MLP vs BSpline vs HEMNA
"""
import torch, torch.nn as nn, torch.nn.functional as F, sys, time
sys.path.insert(0, '.')
from hemna_v3 import VectorizedBSplineLayer, Tier1Linear

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# California Housing betoltese sklearn-bol
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = fetch_california_housing()
X = data.data  # 20640 x 8
y = data.target  # 20640

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizalas
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Torch tensor
X_train = torch.FloatTensor(X_train).to(device)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(device)
X_test = torch.FloatTensor(X_test).to(device)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(device)

print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}, Features: {X_train.shape[1]}")

def count_params(*ms):
    return sum(p.numel() for m in ms for p in m.parameters())

def evaluate(model):
    model.eval()
    with torch.no_grad():
        return F.mse_loss(model(X_test), y_test).item()

epochs = 2000

# Az adat normalizalt -> a BSpline grid a standard normalhoz igazitva [-3, 3]
print(f"\n{'Modell':<35} {'Params':<10} {'Test MSE':<15}")
print("-" * 65)

best_mlp = (None, 1e9, 0)
best_bs = (None, 1e9, 0)
best_hemna = (None, 1e9, 0)

# === MLP ===
print("\n--- MLP ---")
for width in [16, 32, 64, 128, 256]:
    for layers in [1, 2]:
        if layers == 1:
            model = nn.Sequential(nn.Linear(8, width), nn.ReLU(), nn.Linear(width, 1)).to(device)
        else:
            model = nn.Sequential(nn.Linear(8, width), nn.ReLU(), 
                                   nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1)).to(device)
        p = count_params(model)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        for ep in range(epochs):
            opt.zero_grad()
            loss = F.mse_loss(model(X_train), y_train)
            loss.backward()
            opt.step()
        tl = evaluate(model)
        print(f"  {'MLP '+str(layers)+'l w='+str(width):<35} {p:<10} {tl:<15.6f}")
        if tl < best_mlp[2]:
            best_mlp = (f"MLP {layers}l w={width}", p, tl)

# === BSpline ===
print("\n--- BSpline (T3) ---")
for grid in [4, 8, 16, 32, 64]:
    model = VectorizedBSplineLayer(8, 1, grid_min=-3, grid_max=3, grid_size=grid, degree=3).to(device)
    p = count_params(model)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for ep in range(epochs):
        opt.zero_grad()
        loss = F.mse_loss(model(X_train), y_train)
        loss.backward()
        opt.step()
    tl = evaluate(model)
    print(f"  {'BSpline grid='+str(grid):<35} {p:<10} {tl:<15.6f}")
    if tl < best_bs[2]:
        best_bs = (f"BSpline grid={grid}", p, tl)

# === HEMNA T1+T2+T3 ===
class HEM(nn.Module):
    def __init__(self, t1n, t2n, t3n, grid):
        super().__init__()
        self.t1 = Tier1Linear(8, t1n)
        self.t2 = nn.Linear(8, t2n)
        self.t3 = VectorizedBSplineLayer(8, t3n, grid_min=-3, grid_max=3, grid_size=grid)
        self.out = nn.Linear(t1n+t2n+t3n, 1)
    def forward(self, x):
        return self.out(torch.cat([self.t1(x), F.relu(self.t2(x)), self.t3(x)], -1))

print("\n--- HEMNA T1+T2+T3 ---")
for t1n in [4, 8]:
    for t2n in [4, 8]:
        for t3n in [4, 8]:
            for grid in [8, 16]:
                model = HEM(t1n, t2n, t3n, grid).to(device)
                p = count_params(model)
                opt = torch.optim.Adam(model.parameters(), lr=0.01)
                for ep in range(epochs):
                    opt.zero_grad()
                    loss = F.mse_loss(model(X_train), y_train)
                    loss.backward()
                    opt.step()
                tl = evaluate(model)
                print(f"  {'HEMNA T1='+str(t1n)+' T2='+str(t2n)+' T3='+str(t3n)+' g'+str(grid):<35} {p:<10} {tl:<15.6f}")
                if tl < best_hemna[2]:
                    best_hemna = (f"HEMNA T1={t1n}+T2={t2n}+T3={t3n} g{grid}", p, tl)

print("\n=== LEGJOBBAK ===")
for name, data in [("MLP", best_mlp), ("BSpline", best_bs), ("HEMNA", best_hemna)]:
    print(f"  {name}: {data[0]} ({data[1]} param) -> MSE={data[2]:.6f}")
