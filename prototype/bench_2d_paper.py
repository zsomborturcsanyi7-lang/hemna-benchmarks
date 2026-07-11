"""
HEMNA vs MLP vs KAN — 2D function approximation
f(x,y) = sin(x) * cos(y) on [-5, 5] x [-5, 5]

Tobb random seed, parameter-egyenesseg, statisztikailag ervenyes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, time, math
sys.path.insert(0, '.')
from hemna_v3 import VectorizedBSplineLayer, Tier1Linear

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# ============================================================
# Adat: f(x,y) = sin(x) * cos(y) a [-5, 5] tartomanyon
# ============================================================
def generate_data(n=5000, seed=42):
    torch.manual_seed(seed)
    X = torch.rand(n, 2) * 10 - 5  # [-5, 5]
    y = (torch.sin(X[:, 0]) * torch.cos(X[:, 1])).unsqueeze(1)
    return X.to(device), y.to(device)

X_test, y_test = generate_data(2000, 999)

def count_params(*modules):
    return sum(p.numel() for m in modules for p in m.parameters())

def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        return F.mse_loss(model(X), y).item()

# ============================================================
# Egy-szeru KAN implementacio osszehasonlithoz
# ============================================================
class SimpleKANLayer(nn.Module):
    """KAN-szeru reteq: minden bemenetre BSpline, linearis kombinacio."""
    def __init__(self, in_features, out_features, grid_size=8, degree=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Minden bemeneti dimenziora egy BSpline, minden kimenetre sulyozva
        self.splines = nn.ModuleList([
            VectorizedBSplineLayer(1, out_features, grid_min=-5, grid_max=5,
                                    grid_size=grid_size, degree=degree)
            for _ in range(in_features)
        ])
    
    def forward(self, x):
        # x: [B, in_features]
        out = 0
        for i in range(self.in_features):
            out += self.splines[i](x[:, i:i+1])
        return out

# ============================================================
# Benchmark
# ============================================================
seeds = [42, 123, 456]  # tobb seed
configs = []

# == MLP-k ==
for width in [8, 16, 32, 64, 128]:
    for layers in [1, 2]:
        configs.append(('MLP', width, layers, None))

# == BSpline (T3) ==
for grid in [4, 8, 16, 32]:
    configs.append(('BSpline', grid, 1, None))

# == KAN ==
for grid in [4, 8, 16]:
    configs.append(('KAN', grid, 1, None))

# == HEMNA T1+T2+T3 ==
for n_t1 in [4, 8]:
    for n_t2 in [4, 8]:
        for grid in [8, 16]:
            configs.append(('HEMNA', grid, 1, (n_t1, n_t2)))

results = []

for cfg_idx, (model_type, param, n_layers, extra) in enumerate(configs):
    losses = []
    total_params = 0
    
    for seed in seeds:
        torch.manual_seed(seed)
        X_train, y_train = generate_data(5000, seed)
        
        if model_type == 'MLP':
            width = param
            if n_layers == 1:
                model = nn.Sequential(
                    nn.Linear(2, width), nn.ReLU(),
                    nn.Linear(width, 1)
                ).to(device)
            else:
                model = nn.Sequential(
                    nn.Linear(2, width), nn.ReLU(),
                    nn.Linear(width, width), nn.ReLU(),
                    nn.Linear(width, 1)
                ).to(device)
        
        elif model_type == 'BSpline':
            model = VectorizedBSplineLayer(2, 1, grid_min=-5, grid_max=5,
                                            grid_size=param, degree=3).to(device)
        
        elif model_type == 'KAN':
            model = SimpleKANLayer(2, 1, grid_size=param, degree=3).to(device)
        
        elif model_type == 'HEMNA':
            grid_sz = param
            n_t1, n_t2 = extra
            t1_n = n_t1
            t2_n = n_t2
            # T3: egy BSpline a maradek neuronokhoz
            t3_n = max(0, t1_n)  # ugyanannyi T3 mint T1
            t1 = Tier1Linear(2, t1_n).to(device)
            t2 = nn.Linear(2, t2_n).to(device)
            t3 = VectorizedBSplineLayer(2, t3_n, grid_min=-5, grid_max=5,
                                         grid_size=grid_sz, degree=3).to(device)
            out = nn.Linear(t1_n + t2_n + t3_n, 1).to(device)
            
            class HEMNAModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.t1, self.t2, self.t3, self.out = t1, t2, t3, out
                def forward(self, x):
                    h = torch.cat([self.t1(x), F.relu(self.t2(x)), self.t3(x)], dim=-1)
                    return self.out(h)
            model = HEMNAModel().to(device)
        
        total_params = count_params(model)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Training
        for ep in range(2000):
            opt.zero_grad()
            loss = F.mse_loss(model(X_train), y_train)
            loss.backward()
            opt.step()
        
        test_loss = evaluate(model, X_test, y_test)
        losses.append(test_loss)
    
    avg_loss = sum(losses) / len(losses)
    std_loss = (sum((l - avg_loss)**2 for l in losses) / len(losses))**0.5
    
    # Label
    if model_type == 'MLP':
        label = f"MLP {n_layers}layer w={param}"
    elif model_type == 'BSpline':
        label = f"BSpline grid={param}"
    elif model_type == 'KAN':
        label = f"KAN grid={param}"
    elif model_type == 'HEMNA':
        label = f"HEMNA T1={extra[0]}+T2={extra[1]}+T3 gr={param}"
    
    print(f"  [{cfg_idx+1}/{len(configs)}] {label:<35} params={total_params:<8} loss={avg_loss:.6f} +/-{std_loss:.6f}")
    results.append((label, total_params, avg_loss, std_loss))

print(f"\n=== TOP 10 LEGJOBB ===")
results.sort(key=lambda x: x[2])
for i, (label, params, loss, std) in enumerate(results[:10]):
    print(f"  {i+1}. {label:<35} params={params:<8} loss={loss:.6f}")

print(f"\n=== OSSZEFOGLALO ===")
# Legjobb MLP, BSpline, KAN, HEMNA
for cat in ['MLP', 'BSpline', 'KAN', 'HEMNA']:
    best = min([r for r in results if r[0].startswith(cat)], key=lambda x: x[2])
    print(f"  Legjobb {cat}: {best[0]} -> {best[2]:.6f} ({best[1]} param)")
