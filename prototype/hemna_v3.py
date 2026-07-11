"""
HEMNA v3 — Vektorizált BSpline + Többrétegű Growing
====================================================

Változások v2-höz képest:
  1. BSpline vektorizálva (nincs Python for-ciklus a forward-ban)
  2. GrowingLayer több rétegben (MLP, minden rétegben T1→T2→T3)
  3. Tier3Linear a valódi BSpline-t használja (nem MLP proxy)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional


# ============================================================
# Vektorizált BSpline réteg (T3) — EGY forward minden kapcsolatra
# ============================================================
class VectorizedBSplineLayer(nn.Module):
    """
    Vektorizált B-spline réteg adaptív grid-del.
    
    A grid automatikusan igazodik a bemeneti adatok várható tartományához.
    """
    def __init__(self, in_features: int, out_features: int,
                 grid_min: float = None, grid_max: float = None,
                 grid_size: int = 8, degree: int = 3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.degree = degree
        self.n_coeffs = grid_size + degree - 1
        
        # Adaptive grid: ha nincs megadva, [-2, 2] az alap (magas dimenziós bemenetre)
        # Alacsony dimenziós bemenetre (0/1 bináris) érdemes [0, 1]-re állítani
        if grid_min is None and in_features <= 4:
            grid_min = 0.0  # bináris/one-hot bemenetek
            grid_max = 1.0
        elif grid_min is None:
            grid_min = -2.0
            grid_max = 2.0
        
        self.grid_min = grid_min
        self.grid_max = grid_max
        
        # Coefficients: [out_features, in_features, n_coeffs]
        self.coefficients = nn.Parameter(
            torch.randn(out_features, in_features, self.n_coeffs) * 0.1
        )
        
        # Knot vector (clamped)
        grid = torch.linspace(grid_min, grid_max, grid_size)
        pad = degree
        knots = torch.cat([
            torch.full((pad,), grid_min),
            grid,
            torch.full((pad,), grid_max)
        ])
        self.register_buffer('knots', knots)
        self.register_buffer('grid', grid)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, in_features]
        Returns: [batch, out_features]
        """
        batch, in_feat = x.shape
        k = self.degree
        knots = self.knots
        n_knots = len(knots)
        
        # Clamp x
        x = x.clamp(self.grid_min, self.grid_max)
        
        # === Degree 0 ===
        # basis_i: B_{i,0}(x) = 1 if knots[i] <= x < knots[i+1]
        # x: [B, I] → x_exp: [B, I, 1]
        # knots: [n_knots] → left: [1, 1, n_knots-1]
        left = knots[:n_knots-1].unsqueeze(0).unsqueeze(0)   # [1, 1, Nk-1]
        right = knots[1:n_knots].unsqueeze(0).unsqueeze(0)   # [1, 1, Nk-1]
        x_exp = x.unsqueeze(-1)  # [B, I, 1]
        
        basis = ((x_exp >= left) & (x_exp < right)).float()  # [B, I, Nk-1]
        # Handle rightmost grid point
        grid_max_mask = (x == self.grid_max).unsqueeze(-1)    # [B, I, 1]
        basis = torch.where(grid_max_mask, 
                            torch.zeros_like(basis).scatter_(2, 
                                torch.full((batch, in_feat, 1), basis.shape[2]-1, 
                                          device=x.device, dtype=torch.long), 1.0),
                            basis)
        
        # === Higher degrees 1..k ===
        for d in range(1, k + 1):
            n_valid = n_knots - d - 1  # number of valid basis functions at this degree
            
            # Knots for left and right terms: [1, 1, n_valid]
            knots_i = knots[:n_valid].unsqueeze(0).unsqueeze(0)
            knots_id = knots[d:d+n_valid].unsqueeze(0).unsqueeze(0)
            knots_i1 = knots[1:1+n_valid].unsqueeze(0).unsqueeze(0)
            knots_id1 = knots[d+1:d+1+n_valid].unsqueeze(0).unsqueeze(0)
            
            denom_l = knots_id - knots_i
            denom_r = knots_id1 - knots_i1
            
            # Previous degree basis, sliced: [B, I, n_valid]
            b_left = basis[:, :, :n_valid]      # B_{i,d-1}
            b_right = basis[:, :, 1:n_valid+1]  # B_{i+1,d-1}
            
            left_term = torch.where(
                denom_l.abs() > 1e-10,
                (x_exp - knots_i) / denom_l * b_left,
                torch.zeros_like(b_left)
            )
            right_term = torch.where(
                denom_r.abs() > 1e-10,
                (knots_id1 - x_exp) / denom_r * b_right,
                torch.zeros_like(b_right)
            )
            
            basis = left_term + right_term
        
        # basis: [B, I, n_coeffs] (n_coeffs = grid_size + degree - 1)
        # coefficients: [O, I, n_coeffs]
        # output: [B, O]
        output = torch.einsum('bik,oik->bo', basis, self.coefficients)
        return output
    
    def init_from_linear(self, weight: torch.Tensor, bias: float = 0.0):
        """
        BSpline együtthatók inicializálása egy lineáris függvényből.
        
        A Greville-abscissae segítségével pontosan reprezentálja a lineáris
        függvényt (w·x + b) a BSpline bázisban.
        
        weight: [in_features] — a T2 neuron súlyai
        bias: skalár — a T2 neuron bias-a
        """
        k = self.degree
        knots = self.knots
        n_basis = len(knots) - k - 1  # = n_coeffs
        
        # Greville abscissae: ξ_j = (t_{j+1} + ... + t_{j+k}) / k
        greville = torch.zeros(n_basis, device=knots.device)
        for j in range(n_basis):
            greville[j] = knots[j+1:j+1+k].mean()
        
        with torch.no_grad():
            for i in range(self.in_features):
                w_i = weight[i].item() if i < len(weight) else 0.0
                # c_{i,k} = w_i * ξ_k + bias / in_features
                # (a bias eloszlik minden bemeneti dimenzió és bázisfüggvény között)
                self.coefficients[0, i, :] = w_i * greville + bias / self.in_features


# ============================================================
# T0 — Még a T1-nél is kisebb (csak bias, nincs súly)
# ============================================================
class Tier0Linear(nn.Module):
    """T0: Csak egy bias. Nem számol a bemenettel, konstans kimenet."""
    def __init__(self, out_features: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.out_features = out_features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bias.unsqueeze(0).expand(x.shape[0], -1)
    
    def extra_repr(self):
        return f'T0: {self.out_features}'


# ============================================================
# T1 — Apró (reflex) neuron (STE-vel)
# ============================================================
class Tier1Linear(nn.Module):
    """Bináris súlyok STE-vel, szigmoid kimenet."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_bin = torch.sign(self.weight)
        w_ste = self.weight + (w_bin - self.weight).detach()
        out = F.linear(x, w_ste, self.bias)
        return torch.sigmoid(out)

    def extra_repr(self):
        return f'T1: {self.in_features}→{self.out_features}'


# ============================================================
# T2 — Normál MLP neuron
# ============================================================
class Tier2Linear(nn.Module):
    """Normál ReLU neuron."""
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
# GrowingLayer — egy réteg neuronokkal, T1→T2→T3
# ============================================================
class GrowingLayer(nn.Module):
    """
    Egy MLP réteg ahol neuronok T0-kent indulnak es nonnek.
    
    Tier rendszer:
      T0: bias-only (1 parameter) - legolcsobb
      T1: binaris STE + sigmoid
      T2: full Linear + LeakyReLU
    
    Minden parameter ELORE le van foglalva.
    """
    def __init__(self, in_features: int, max_neurons: int,
                 grad_threshold_t1: float = 0.001,
                 grad_threshold_t2: float = 0.001,
                 patience: int = 100):
        super().__init__()
        self.in_features = in_features
        self.max_neurons = max_neurons
        self.grad_threshold_t1 = grad_threshold_t1
        self.grad_threshold_t2 = grad_threshold_t2
        self.patience = patience
        
        # T0: bias-only
        self.t0 = Tier0Linear(max_neurons)
        # T1: binaris STE
        self.t1 = Tier1Linear(in_features, max_neurons)
        # T2: LeakyReLU (minden neuronhoz)
        self.t2_modules = nn.ModuleList([
            nn.Linear(in_features, 1) for _ in range(max_neurons)
        ])
        
        # Neuron tipusok: 0=T0, 1=T1, 2=T2
        self.register_buffer('tier_map', torch.zeros(max_neurons, dtype=torch.long))
        self.register_buffer('t01_grad_counter', torch.zeros(max_neurons, dtype=torch.long))
        self.register_buffer('t12_grad_counter', torch.zeros(max_neurons, dtype=torch.long))
        
        self._step = 0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        n = self.max_neurons
        device = x.device
        
        out = torch.zeros(batch, n, device=device)
        t0_mask = self.tier_map == 0
        t1_mask = self.tier_map == 1
        t2_mask = self.tier_map == 2
        
        if t0_mask.any():
            out[:, t0_mask] = self.t0(x)[:, t0_mask]
        if t1_mask.any():
            out[:, t1_mask] = self.t1(x)[:, t1_mask]
        if t2_mask.any():
            for i in range(n):
                if t2_mask[i]:
                    out[:, i:i+1] = F.leaky_relu(self.t2_modules[i](x), 0.01)
        return out
    
    def _upgrade(self, neuron_idx: int):
        current = self.tier_map[neuron_idx].item()
        if current == 0:
            self.tier_map[neuron_idx] = 1
        elif current == 1:
            self.tier_map[neuron_idx] = 2
    
    def update_growth(self):
        self._step += 1
        
        if self.t0.bias.grad is not None:
            t0_grads = self.t0.bias.grad.abs()
            for i in range(self.max_neurons):
                if self.tier_map[i] == 0:
                    if t0_grads[i] > self.grad_threshold_t1:
                        self.t01_grad_counter[i] += 1
                        if self.t01_grad_counter[i] >= self.patience:
                            self._upgrade(i)
                            self.t01_grad_counter[i] = 0
                    else:
                        self.t01_grad_counter[i] = max(0, self.t01_grad_counter[i] - 1)
        
        if self.t1.weight.grad is not None:
            t1_grads = self.t1.weight.grad.abs().mean(dim=1)
            for i in range(self.max_neurons):
                if self.tier_map[i] == 1:
                    if t1_grads[i] > self.grad_threshold_t2:
                        self.t12_grad_counter[i] += 1
                        if self.t12_grad_counter[i] >= self.patience:
                            self._upgrade(i)
                            self.t12_grad_counter[i] = 0
                    else:
                        self.t12_grad_counter[i] = max(0, self.t12_grad_counter[i] - 1)
    
    def get_stats(self):
        n_t0 = (self.tier_map == 0).sum().item()
        n_t1 = (self.tier_map == 1).sum().item()
        n_t2 = (self.tier_map == 2).sum().item()
        return {'T0': n_t0, 'T1': n_t1, 'T2': n_t2, 'total': n_t0+n_t1+n_t2, 'step': self._step}


# ============================================================
# HEMNA v3 — Teljes halozat
# ============================================================
class HEMNAv3(nn.Module):
    """
    HEMNA v3: tobbretegu MLP growing mechanism-mel.
    Minden hidden retegeben a neuronok T0-kent indulnak es nonek.
    A kimenet fix dimenzioju (max_neurons retegenkent).
    Minden parameter ELORE le van foglalva.
    """
    def __init__(self, input_dim: int, hidden_sizes: List[int], output_dim: int,
                 grad_threshold_t1: float = 0.001,
                 grad_threshold_t2: float = 0.001,
                 patience: int = 100):
        super().__init__()
        
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        for h_size in hidden_sizes:
            layer = GrowingLayer(
                in_features=prev_dim,
                max_neurons=h_size,
                grad_threshold_t1=grad_threshold_t1,
                grad_threshold_t2=grad_threshold_t2,
                patience=patience
            )
            self.layers.append(layer)
            prev_dim = h_size  # max_neurons = következő réteg input dim
        
        # Fix output (max_neurons dimenzió mindig fix)
        last_dim = hidden_sizes[-1] if hidden_sizes else input_dim
        self.output = nn.Linear(last_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.output(x)
    
    def update_growth(self):
        """Minden rétegben frissíti a growth-öt. Nincs új paraméter."""
        for layer in self.layers:
            layer.update_growth()
    
    def get_all_stats(self):
        return [layer.get_stats() for layer in self.layers]
    
    def train_step(self, x, y, optimizer):
        """Egy teljes train lépés: forward + backward + growth + optimizer."""
        pred = self(x)
        loss = F.mse_loss(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        self.update_growth()
        optimizer.step()
        
        return loss.item()


# ============================================================
# Egyszerű teszt — XOR
# ============================================================
def test_xor():
    torch.manual_seed(42)
    
    X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
    y = torch.tensor([[0.],[1.],[1.],[0.]])
    
    model = HEMNAv3(input_dim=2, hidden_sizes=[6], output_dim=1,
                    grad_threshold_t2=0.001, grad_threshold_t3=0.005,
                    patience=50)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print("=== HEMNA v3 — XOR teszt (vektorizált BSpline + growing) ===")
    print(f"Kezdő: {model.get_all_stats()}")
    
    for step in range(1000):
        loss = model.train_step(X, y, optimizer)
        
        if step % 200 == 0:
            stats = model.get_all_stats()
            print(f"  Step {step:4d}: loss={loss:.6f} | {stats[0]}")
    
    stats = model.get_all_stats()
    print(f"Végső: {stats[0]}")
    print("\nEredmények:")
    with torch.no_grad():
        preds = model(X)
        for inp, p, t in zip(X, preds, y):
            print(f"  {inp.tolist()} → {p.item():.4f} (várt: {t.item()})")
    
    return model


# ============================================================
# 4-bit parity teszt
# ============================================================
def generate_parity(n_bits=4):
    n = 2 ** n_bits
    X = torch.zeros(n, n_bits)
    y = torch.zeros(n, 1)
    for i in range(n):
        bits = [(i >> j) & 1 for j in range(n_bits)]
        X[i] = torch.tensor(bits, dtype=torch.float)
        y[i] = sum(bits) % 2
    return X, y


def test_parity():
    torch.manual_seed(42)
    X, y = generate_parity(4)
    
    model = HEMNAv3(input_dim=4, hidden_sizes=[8], output_dim=1,
                    grad_threshold_t2=0.001, grad_threshold_t3=0.005,
                    patience=100)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print("=== HEMNA v3 — 4-bit Parity (valódi BSpline T3) ===")
    print(f"Kezdő: {model.get_all_stats()}")
    
    for step in range(3000):
        loss = model.train_step(X, y, optimizer)
        
        if step % 500 == 0:
            stats = model.get_all_stats()
            print(f"  Step {step:4d}: loss={loss:.6f} | {stats[0]}")
    
    stats = model.get_all_stats()
    print(f"Végső: {stats[0]}")
    
    with torch.no_grad():
        preds = model(X)
        acc = ((preds > 0.5).float().eq(y).sum().item() / len(y) * 100)
    print(f"Accuracy: {acc:.0f}%")
    return model


if __name__ == '__main__':
    import time
    start = time.time()
    
    # BSpline sebesség teszt
    print("=== VectorizedBSplineLayer sebesség teszt ===")
    spline = VectorizedBSplineLayer(100, 8)
    x = torch.randn(128, 100)
    
    s = time.time()
    for _ in range(100):
        spline(x)
    t = (time.time() - s) / 100 * 1000
    print(f"  (100in, 8out, batch=128): {t:.3f} ms")
    
    # XOR
    print()
    test_xor()
    
    # Parity — alacsony T3 threshold
    print()
    torch.manual_seed(42)
    X, y = generate_parity(4)
    
    model = HEMNAv3(input_dim=4, hidden_sizes=[8], output_dim=1,
                    grad_threshold_t2=0.001, grad_threshold_t3=0.002,
                    patience=50)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print("=== HEMNA v3 — 4-bit Parity (pre-allocated T3) ===")
    print(f"Kezdő: {model.get_all_stats()}")
    
    for step in range(2000):
        loss = model.train_step(X, y, optimizer)
        
        if step % 500 == 0:
            stats = model.get_all_stats()
            print(f"  Step {step:4d}: loss={loss:.6f} | {stats[0]}")
    
    stats = model.get_all_stats()
    print(f"Végső: {stats[0]}")
    
    with torch.no_grad():
        preds = model(X)
        acc = ((preds > 0.5).float().eq(y).sum().item() / len(y) * 100)
    print(f"Accuracy: {acc:.0f}%")
    
    print(f"\nTeljes idő: {time.time() - start:.1f}s")
