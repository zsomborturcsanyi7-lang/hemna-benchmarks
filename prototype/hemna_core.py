"""
HEMNA — Heterogén Multi-Skálájú Neurális Architektúra
PROTOTÍPUS V1.0

Egy MLP ahol neuronok különböző típusúak lehetnek:
  T1: Apró (reflex, bináris)
  T2: Kis (normál MLP, ReLU)
  T3: Nagy (spline-alapú, intelligens)

Fő újítás: Growing mechanism — neuronok T1 indul, gradient alapján nő.

Szerző: János (14) + Hermes Agent
Dátum: 2026.06.20
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Optional, Literal

# ============================================================
# T1 — Apró (reflex) neuron
# ============================================================
class Tier1Linear(nn.Module):
    """
    Bináris súlyok, bináris kimenet.
    y = step(Σ w_i · x_i)
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Bináris súlyok: -1 vagy +1
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Straight-Through Estimator: sign in forward, identity in backward
        w_bin = torch.sign(self.weight)
        w_ste = self.weight + (w_bin - self.weight).detach()
        out = F.linear(x, w_ste, self.bias)
        # Folytonos kimenet [0,1] — traininghez gradient átfolyik
        return torch.sigmoid(out)

    def extra_repr(self):
        return f'T1: {self.in_features}→{self.out_features}'


# ============================================================
# T2 — Kis (normál MLP) neuron
# ============================================================
class Tier2Linear(nn.Module):
    """
    Normál MLP neuron.
    y = ReLU(Wx + b)
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.linear(x))

    def extra_repr(self):
        return f'T2: {self.in_features}→{self.out_features}'


# ============================================================
# T3 — Nagy (spline-alapú) neuron
# ============================================================
class BSpline(nn.Module):
    """Egy egyszerű B-spline implementáció.
    
    Cox-de Boor rekurzió B-spline bázisfüggvényekhez.
    Grid: torch.linspace(-2, 2, grid_size)
    """
    def __init__(self, grid_size: int = 8, degree: int = 3):
        super().__init__()
        self.grid_size = grid_size
        self.degree = degree
        n_coeffs = grid_size + degree - 1
        self.coefficients = nn.Parameter(torch.randn(n_coeffs) * 0.1)
        grid = torch.linspace(-2.0, 2.0, grid_size)
        self.register_buffer('grid', grid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cox-de Boor rekurzió
        # degree=0 eset
        n = self.grid_size - 1
        k = self.degree
        
        # Kiterjesztett grid (clamp miatt)
        # Egyszerűbb: használjunk explicit basis számolást
        batch = x.shape[0]
        device = x.device
        
        # Cox-de Boor
        # degree=0
        basis = torch.zeros(batch, self.grid_size - 1, device=device)
        for i in range(self.grid_size - 1):
            mask = (x >= self.grid[i]) & (x < self.grid[i + 1])
            if i < basis.shape[1]:
                basis[mask, i] = 1.0
        # Utolsó grid pontot is vegyük bele
        mask = x == self.grid[-1]
        if mask.any() and self.grid_size - 2 >= 0:
            basis[mask, -1] = 1.0
        
        # Magasabb fokok
        n_basis = basis.shape[1]
        for d in range(1, k + 1):
            new_basis = torch.zeros_like(basis)
            for i in range(n_basis - d):
                # Left term
                denom_l = self.grid[i + d] - self.grid[i]
                left = torch.zeros(batch, device=device)
                if denom_l > 0:
                    left = (x - self.grid[i]) / denom_l * basis[:, i]
                
                # Right term
                denom_r = self.grid[i + d + 1] - self.grid[i + 1]
                right = torch.zeros(batch, device=device)
                if denom_r > 0:
                    right = (self.grid[i + d + 1] - x) / denom_r * basis[:, i + 1]
                
                new_basis[:, i] = left + right
            basis = new_basis
        
        # Összegzés koefficiensekkel
        # Csak az első n_coeffs bázist használjuk
        n_coeffs = self.grid_size + k - 1
        coeffs = self.coefficients[:min(n_coeffs, basis.shape[1])]
        result = basis[:, :len(coeffs)] @ coeffs
        return result


class Tier3Linear(nn.Module):
    """
    Nagy neuron — minden kapcsolaton egy B-spline.
    y = Σ spline_i(x_i)
    """
    def __init__(self, in_features: int, out_features: int, grid_size: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        
        # Minden (out, in) kapcsolathoz egy B-spline
        self.splines = nn.ModuleList([
            nn.ModuleList([BSpline(grid_size=grid_size) for _ in range(in_features)])
            for _ in range(out_features)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        outputs = []
        for out_idx in range(self.out_features):
            out_val = torch.zeros(batch_size, device=x.device)
            for in_idx in range(self.in_features):
                out_val += self.splines[out_idx][in_idx](x[:, in_idx])
            outputs.append(out_val.unsqueeze(1))
        return torch.cat(outputs, dim=1)

    def extra_repr(self):
        return f'T3: {self.in_features}→{self.out_features}'


# ============================================================
# HEMNA Réteg — vegyes típusú neuronok
# ============================================================
class HEMNALayer(nn.Module):
    """
    Egy réteg a HEMNA hálóban.
    Tartalmaz T1, T2 és T3 neuronokat.
    """
    def __init__(self, in_features: int, out_features: int,
                 tier_ratio: dict = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        if tier_ratio is None:
            tier_ratio = {'T1': 0.3, 'T2': 0.5, 'T3': 0.2}
        
        n_t1 = max(1, int(out_features * tier_ratio['T1']))
        n_t2 = max(1, int(out_features * tier_ratio['T2']))
        n_t3 = out_features - n_t1 - n_t2
        
        self.T1 = Tier1Linear(in_features, n_t1) if n_t1 > 0 else None
        self.T2 = Tier2Linear(in_features, n_t2) if n_t2 > 0 else None
        self.T3 = Tier3Linear(in_features, n_t3) if n_t3 > 0 else None
        
        # Tároljuk a tényleges neuron számokat
        self.n_t1 = n_t1
        self.n_t2 = n_t2
        self.n_t3 = n_t3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        if self.T1 is not None:
            outputs.append(self.T1(x))
        if self.T2 is not None:
            outputs.append(self.T2(x))
        if self.T3 is not None:
            outputs.append(self.T3(x))
        return torch.cat(outputs, dim=-1)


# ============================================================
# Teljes HEMNA hálózat
# ============================================================
class HEMNA(nn.Module):
    """
    HEMNA architektúra.
    
    Example:
        model = HEMNA(input_dim=10, hidden_dims=[20, 15], output_dim=2,
                       layer_ratios=[{'T1':0.3,'T2':0.5,'T3':0.2},
                                     {'T1':0.2,'T2':0.4,'T3':0.4}])
    """
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 layer_ratios: Optional[List[dict]] = None):
        super().__init__()
        
        if layer_ratios is None:
            layer_ratios = [{'T1': 0.3, 'T2': 0.5, 'T3': 0.2}] * len(hidden_dims)
        
        assert len(layer_ratios) == len(hidden_dims), \
            f"{len(layer_ratios)} ratios for {len(hidden_dims)} hidden layers"
        
        layers = []
        prev_dim = input_dim
        
        for i, (h_dim, ratio) in enumerate(zip(hidden_dims, layer_ratios)):
            layers.append(HEMNALayer(prev_dim, h_dim, tier_ratio=ratio))
            prev_dim = h_dim
        
        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)
    
    def get_tier_stats(self):
        """Visszaadja a neuron típusok statisztikáit."""
        stats = []
        for i, layer in enumerate(self.hidden_layers):
            stats.append({
                'layer': i,
                'T1': layer.n_t1,
                'T2': layer.n_t2,
                'T3': layer.n_t3,
                'total': layer.out_features,
                'T1_pct': f"{100*layer.n_t1/layer.out_features:.0f}%",
                'T2_pct': f"{100*layer.n_t2/layer.out_features:.0f}%",
                'T3_pct': f"{100*layer.n_t3/layer.out_features:.0f}%",
            })
        return stats


# ============================================================
# Teszt — XOR feladat
# ============================================================
def test_xor():
    """XOR teszt: HEMNA vs Standard MLP."""
    torch.manual_seed(42)
    
    # Adatok
    X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
    y = torch.tensor([[0.],[1.],[1.,],[0.]])
    
    # HEMNA
    hemna = HEMNA(input_dim=2, hidden_dims=[8], output_dim=1,
                  layer_ratios=[{'T1':0.25,'T2':0.5,'T3':0.25}])
    optimizer = torch.optim.Adam(hemna.parameters(), lr=0.01)
    
    print("=== HEMNA XOR teszt ===")
    print("Réteg statisztika:")
    for s in hemna.get_tier_stats():
        print(f"  Layer {s['layer']}: T1={s['T1_pct']} T2={s['T2_pct']} T3={s['T3_pct']} ({s['total']} neuron)")
    
    losses = []
    for epoch in range(1000):
        pred = hemna(X)
        loss = F.mse_loss(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if epoch % 200 == 0:
            print(f"  Epoch {epoch}: loss = {loss.item():.6f}")
    
    print(f"\nVégső loss: {losses[-1]:.6f}")
    print("Kimenetek:")
    with torch.no_grad():
        preds = hemna(X)
        for i, (inp, p, t) in enumerate(zip(X, preds, y)):
            print(f"  {inp.tolist()} → {p.item():.3f} (várt: {t.item()})")
    
    return losses


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    losses = test_xor()
    print("\n✅ Prototípus kész. A growing mechanism implementáció következik.")
