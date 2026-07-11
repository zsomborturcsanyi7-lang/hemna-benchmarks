"""
HEMNA — Growing T2→T3 Teszt: 4-bit Parity
Gyors T3 proxy (MLP), hogy a growing mechanism-t teszteljük.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, '.')
from hemna_growing import GrowingLayer


# Gyors T3 proxy — a BSpline helyett, ami 1000× lassabb
class Tier3Fast(nn.Module):
    """T3 proxy: egy kis MLP a B-spline helyett."""
    def __init__(self, in_features, out_features, hidden=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features)
        )
    def forward(self, x):
        return self.net(x)


class AdaptiveOutput(nn.Module):
    def __init__(self, initial_in: int, out: int = 1):
        super().__init__()
        self.initial_in = initial_in
        self.out = out
        self.weight = nn.Parameter(torch.randn(out, initial_in) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out))
        self.extra_weights = nn.ParameterList()
    
    def forward(self, x):
        in_dim = x.shape[-1]
        if in_dim <= self.initial_in:
            return F.linear(x, self.weight[:, :in_dim], self.bias)
        else:
            out = F.linear(x[..., :self.initial_in], self.weight, self.bias)
            extra_start = self.initial_in
            for w in self.extra_weights:
                if extra_start + w.shape[1] <= in_dim:
                    out = out + F.linear(x[..., extra_start:extra_start + w.shape[1]], w, None)
                    extra_start += w.shape[1]
            return out
    
    def extend(self, new_dims: int):
        w = nn.Parameter(torch.zeros(self.out, new_dims))
        self.extra_weights.append(w)
        return w


def generate_parity(n_bits=4):
    n = 2 ** n_bits
    X = torch.zeros(n, n_bits)
    y = torch.zeros(n, 1)
    for i in range(n):
        bits = [(i >> j) & 1 for j in range(n_bits)]
        X[i] = torch.tensor(bits, dtype=torch.float)
        y[i] = sum(bits) % 2
    return X, y


class GrowingLayerFastT3(GrowingLayer):
    """GrowingLayer T3 fast proxy-val a BSpline helyett."""
    def _upgrade_neuron(self, neuron_idx: int):
        current_tier = self.tier_map[neuron_idx].item()
        if current_tier == 0:
            new_layer = nn.Linear(self.in_features, 1)  # T1→T2: sima linear
            self.t2_layers.append(new_layer)
            self._t2_neuron_idx.append(neuron_idx)
            self.tier_map[neuron_idx] = 1
            print(f"  ↑ Neuron {neuron_idx}: T1 → T2 (step {self._step})")
        elif current_tier == 1:
            new_layer = Tier3Fast(self.in_features, 1)  # T2→T3: MLP proxy
            self.t3_layers.append(new_layer)
            self.tier_map[neuron_idx] = 2
            print(f"  ↑ Neuron {neuron_idx}: T2 → T3 (step {self._step})")
    
    def forward(self, x, track_grads=False):
        batch_size = x.shape[0]
        outputs = []
        t1_out = self.neuron_layer(x)
        outputs.append(t1_out)
        
        t2_out = torch.zeros(batch_size, len(self.t2_layers), device=x.device)
        for i, layer in enumerate(self.t2_layers):
            t2_out[:, i:i+1] = layer(x)
        outputs.append(t2_out)
        
        t3_out = torch.zeros(batch_size, len(self.t3_layers), device=x.device)
        for i, layer in enumerate(self.t3_layers):
            t3_out[:, i:i+1] = layer(x)
        outputs.append(t3_out)
        
        return torch.cat(outputs, dim=-1)
    
    def update_growth(self):
        """T2→T3 fix: t2_layers now stores nn.Linear, check its weight.grad"""
        self._step += 1
        
        # T1 gradient check
        if self.neuron_layer.weight.grad is not None:
            grads = self.neuron_layer.weight.grad.abs().mean(dim=1)
            for i in range(self.max_neurons):
                if self.tier_map[i] == 0:
                    if grads[i] > self.grad_threshold_t2:
                        self.grad_counter[i] += 1
                        if self.grad_counter[i] >= self.patience:
                            self._upgrade_neuron(i)
                            self.grad_counter[i] = 0
                    else:
                        self.grad_counter[i] = max(0, self.grad_counter[i] - 1)
        
        # T2 → T3
        for i, layer in enumerate(self.t2_layers):
            if layer.weight.grad is not None:
                grad_norm = layer.weight.grad.abs().mean().item()
                neuron_idx = self._t2_neuron_idx[i]
                if grad_norm > self.grad_threshold_t3:
                    self.t3_grad_counter[neuron_idx] += 1
                    if self.t3_grad_counter[neuron_idx] >= self.patience:
                        self._upgrade_neuron(neuron_idx)
                        self.t3_grad_counter[neuron_idx] = 0
                else:
                    self.t3_grad_counter[neuron_idx] = max(0, self.t3_grad_counter[neuron_idx] - 1)


def train_growing(max_neurons, th_t2, th_t3, label, n_steps=6000):
    X, y = generate_parity(4)
    
    layer = GrowingLayerFastT3(in_features=4, max_neurons=max_neurons,
                                grad_threshold_t2=th_t2,
                                grad_threshold_t3=th_t3,
                                patience=100)
    output = AdaptiveOutput(initial_in=max_neurons)
    optimizer = torch.optim.Adam(
        list(layer.parameters()) + list(output.parameters()), lr=0.01)
    
    print(f"--- {label} ---")
    print(f"Kezdő: {layer.get_stats()}")
    
    growth_events = []
    
    for step in range(n_steps):
        h = layer(X, track_grads=True)
        pred = output(h)
        loss = F.mse_loss(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        
        old_t2 = len(layer.t2_layers)
        old_t3 = len(layer.t3_layers)
        layer.update_growth()
        
        new_t2 = len(layer.t2_layers)
        new_t3 = len(layer.t3_layers)
        if new_t2 > old_t2:
            w = output.extend(1)
            optimizer.add_param_group({'params': [w]})
            growth_events.append((step, "T1→T2"))
        if new_t3 > old_t3:
            w = output.extend(1)
            optimizer.add_param_group({'params': [w]})
            growth_events.append((step, "T2→T3"))
        
        optimizer.step()
        
        if step % 1000 == 0:
            stats = layer.get_stats()
            print(f"  Step {step:4d}: loss={loss.item():.6f} | {stats}")
    
    stats = layer.get_stats()
    print(f"Végső: {stats}")
    
    with torch.no_grad():
        preds = output(layer(X))
        acc = ((preds > 0.5).float().eq(y).sum().item() / len(y) * 100)
    print(f"Accuracy: {acc:.0f}%")
    
    for g in growth_events:
        print(f"  {g[0]:4d}: {g[1]}")
    if not growth_events:
        print("  ❌ NEM nőtt egy neuron sem")
    print()
    
    return stats, acc, growth_events


def test_growing():
    torch.manual_seed(42)
    results = []
    
    # A: 8 T1, alacsony T3 threshold
    s, a, g = train_growing(8, 0.001, 0.01, "A: 8 T1, thT2=0.001, thT3=0.01")
    results.append(("A (thT2=0.001, thT3=0.01)", s, a, g))
    
    # B: 8 T1, nagyon alacsony T3
    s, a, g = train_growing(8, 0.001, 0.005, "B: 8 T1, thT2=0.001, thT3=0.005")
    results.append(("B (thT2=0.001, thT3=0.005)", s, a, g))
    
    # C: 16 T1, hogy több growth legyen
    s, a, g = train_growing(16, 0.001, 0.005, "C: 16 T1, thT2=0.001, thT3=0.005")
    results.append(("C (16 T1, thT2=0.001, thT3=0.005)", s, a, g))
    
    print("=== ÖSSZEGZÉS ===")
    for name, stats, acc, growths in results:
        g_str = ' | '.join([f"{s}:{t}" for s,t in growths]) if growths else "NO growth ❌"
        print(f"  {name}: T1={stats['T1']} T2={stats['T2']} T3={stats['T3']} — {acc:.0f}% — {g_str}")


if __name__ == '__main__':
    test_growing()
