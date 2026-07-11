"""
HEMNA — Growing Mechanism Teszt: 4-bit Parity

A 4-bites parity feladat sokkal nehezebb mint az XOR.
T1 neuronok most már STE-vel + sigmoid-dal tanulnak → growth működik!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, '.')
from hemna_growing import GrowingLayer


class AdaptiveOutput(nn.Module):
    """Output réteg ami nő a GrowingLayer-rel együtt."""
    def __init__(self, initial_in: int, out: int = 1):
        super().__init__()
        self.initial_in = initial_in
        self.out = out
        self.weight = nn.Parameter(torch.randn(out, initial_in) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out))
        # Extra súlyok a megnőtt dimenziókhoz
        self.extra_weights = nn.ParameterList()
    
    def forward(self, x):
        in_dim = x.shape[-1]
        if in_dim <= self.initial_in:
            return F.linear(x, self.weight[:, :in_dim], self.bias)
        else:
            # Alap rész
            out = F.linear(x[..., :self.initial_in], self.weight, self.bias)
            # Extra rész — összeadás
            extra_start = self.initial_in
            for i, w in enumerate(self.extra_weights):
                if extra_start + w.shape[1] <= in_dim:
                    out = out + F.linear(x[..., extra_start:extra_start + w.shape[1]], w, None)
                    extra_start += w.shape[1]
            return out
    
    def extend(self, new_dims: int):
        """Új dimenziók hozzáadása (amikor neuron nő)."""
        w = nn.Parameter(torch.zeros(self.out, new_dims))
        self.extra_weights.append(w)


def generate_parity(n_bits=4):
    """Generál parity feladatot n_bits bemenettel."""
    n = 2 ** n_bits
    X = torch.zeros(n, n_bits)
    y = torch.zeros(n, 1)
    for i in range(n):
        bits = [(i >> j) & 1 for j in range(n_bits)]
        X[i] = torch.tensor(bits, dtype=torch.float)
        y[i] = sum(bits) % 2
    return X, y


def train_growing(max_neurons, threshold, label, n_steps=3000):
    """Egy konfiguráció tanítása."""
    X, y = generate_parity(4)
    
    layer = GrowingLayer(in_features=4, max_neurons=max_neurons,
                         grad_threshold_t2=threshold, patience=100)
    output = AdaptiveOutput(initial_in=max_neurons)
    params = list(layer.parameters()) + list(output.parameters())
    optimizer = torch.optim.Adam(params, lr=0.01)
    
    print(f"--- {label} ---")
    print(f"Kezdő: {layer.get_stats()}")
    
    growth_events = []
    
    for step in range(n_steps):
        h = layer(X, track_grads=True)
        pred = output(h)
        loss = F.mse_loss(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Growth előtti dimenzió
        old_t2 = len(layer.t2_layers)
        old_t3 = len(layer.t3_layers)
        
        layer.update_growth()
        
        # Ha nőtt, bővítsük az output réteget
        new_t2 = len(layer.t2_layers)
        new_t3 = len(layer.t3_layers)
        if new_t2 > old_t2:
            output.extend(1)  # egy T2 neuron
            # Új paraméter az optimizerhez
            optimizer.add_param_group({'params': [output.extra_weights[-1]]})
            growth_events.append((step, f"T1→T2"))
        if new_t3 > old_t3:
            output.extend(1)
            optimizer.add_param_group({'params': [output.extra_weights[-1]]})
            growth_events.append((step, f"T2→T3"))
        
        optimizer.step()
        
        if step % 500 == 0:
            stats = layer.get_stats()
            print(f"  Step {step:4d}: loss={loss.item():.6f} | {stats}")
    
    stats = layer.get_stats()
    print(f"Végső: {stats}")
    
    # Accuracy
    with torch.no_grad():
        preds = output(layer(X))
        acc = ((preds > 0.5).float().eq(y).sum().item() / len(y) * 100)
    print(f"Accuracy: {acc:.0f}%")
    
    if growth_events:
        print(f"Növekedések: {growth_events}")
    else:
        print("⚠️  NEM nőtt egy neuron sem")
    print()
    
    return stats, acc, growth_events


def test_growing_parity():
    torch.manual_seed(42)
    
    results = []
    
    # Konfig A: 8 T1, threshold=0.005
    s, a, g = train_growing(8, 0.005, "Konfig A: 8 T1, th=0.005")
    results.append(("A (8 T1, th=0.005)", s, a, g))
    
    # Konfig B: 4 T1, threshold=0.005 (kevés neuron)
    s, a, g = train_growing(4, 0.005, "Konfig B: 4 T1, th=0.005")
    results.append(("B (4 T1, th=0.005)", s, a, g))
    
    # Konfig C: 8 T1, threshold=0.001 (alacsonyabb küszöb)
    s, a, g = train_growing(8, 0.001, "Konfig C: 8 T1, th=0.001")
    results.append(("C (8 T1, th=0.001)", s, a, g))
    
    # Konfig D: 16 T1, threshold=0.01 (magasabb küszöb, több neuron)
    s, a, g = train_growing(16, 0.01, "Konfig D: 16 T1, th=0.01")
    results.append(("D (16 T1, th=0.01)", s, a, g))
    
    print("=== ÖSSZEGZÉS ===")
    for name, stats, acc, growths in results:
        growth_str = f"{len(growths)} growth" if growths else "NO growth ❌"
        t2_t3 = f"T2={stats['T2']} T3={stats['T3']}"
        print(f"  {name}: T1={stats['T1']} {t2_t3} — {acc:.0f}% — {growth_str}")


if __name__ == '__main__':
    test_growing_parity()
