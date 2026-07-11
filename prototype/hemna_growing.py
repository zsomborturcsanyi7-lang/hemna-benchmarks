"""
HEMNA — Growing Mechanism Prototípus

A neuronok T1-ként indulnak, és gradient threshold alapján nőnek.
T1 → T2 → T3

Ez a prototípus egy egyszerű feladaton bizonyítja hogy a mechanizmus működik.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from hemna_core import HEMNA, Tier1Linear, Tier2Linear, Tier3Linear


class GrowingLayer(nn.Module):
    """
    HEMNA réteg ahol neuronok T1-ként indulnak és nőnek.
    
    A neuronok kezdetben mind T1 (apró, bináris).
    Tanulás közben a gradient norm alapján nőnek T2-re, majd T3-ra.
    """
    def __init__(self, in_features: int, max_neurons: int,
                 grad_threshold_t2: float = 0.01,
                 grad_threshold_t3: float = 0.05,
                 patience: int = 50):
        super().__init__()
        self.in_features = in_features
        self.max_neurons = max_neurons
        self.grad_threshold_t2 = grad_threshold_t2
        self.grad_threshold_t3 = grad_threshold_t3
        self.patience = patience  # hány lépés után nő
        
        # Minden neuron kezdetben T1
        self.neuron_layer = Tier1Linear(in_features, max_neurons)
        
        # Neuron típusok követése (0=T1, 1=T2, 2=T3)
        self.register_buffer('tier_map', torch.zeros(max_neurons, dtype=torch.long))
        
        # Gradient history
        self.register_buffer('grad_history', torch.zeros(max_neurons))
        self.register_buffer('grad_counter', torch.zeros(max_neurons, dtype=torch.long))
        
        # T2 és T3 rétegek (lazy init — csak ha kell)
        self.t2_layers = nn.ModuleList()
        self.t3_layers = nn.ModuleList()
        
        # T2 layer → eredeti neuron index mapping
        self._t2_neuron_idx = []  # t2_layers index → neuron index
        
        # T2→T3 patience counter
        self.register_buffer('t3_grad_counter', torch.zeros(max_neurons, dtype=torch.long))
        
        self._step = 0
    
    def _upgrade_neuron(self, neuron_idx: int):
        """Egy neuront feljebb léptet."""
        current_tier = self.tier_map[neuron_idx].item()
        
        if current_tier == 0:
            # T1 → T2
            new_layer = Tier2Linear(self.in_features, 1)
            self.t2_layers.append(new_layer)
            self._t2_neuron_idx.append(neuron_idx)  # track mapping
            self.tier_map[neuron_idx] = 1
            print(f"  ↑ Neuron {neuron_idx}: T1 → T2 (step {self._step})")
            
        elif current_tier == 1:
            # T2 → T3
            new_layer = Tier3Linear(self.in_features, 1)
            self.t3_layers.append(new_layer)
            self.tier_map[neuron_idx] = 2
            print(f"  ↑ Neuron {neuron_idx}: T2 → T3 (step {self._step})")
    
    def forward(self, x: torch.Tensor, track_grads: bool = False) -> torch.Tensor:
        """
        Forward pass. Ha track_grads=True, a gradient-eket is követi
        a growing mechanism számára.
        """
        batch_size = x.shape[0]
        outputs = []
        
        # T1 neuronok — mindig az alaplayer
        t1_out = self.neuron_layer(x)  # [batch, max_neurons]
        outputs.append(t1_out)
        
        # T2 neuronok
        t2_out = torch.zeros(batch_size, len(self.t2_layers), device=x.device)
        for i, layer in enumerate(self.t2_layers):
            t2_out[:, i:i+1] = layer(x)
        outputs.append(t2_out)
        
        # T3 neuronok
        t3_out = torch.zeros(batch_size, len(self.t3_layers), device=x.device)
        for i, layer in enumerate(self.t3_layers):
            t3_out[:, i:i+1] = layer(x)
        outputs.append(t3_out)
        
        return torch.cat(outputs, dim=-1)
    
    def update_growth(self):
        """Egy lépés a growing mechanism-ben."""
        self._step += 1
        
        # Minden T1 neuron gradient-jét nézzük
        if self.neuron_layer.weight.grad is not None:
            grads = self.neuron_layer.weight.grad.abs().mean(dim=1)
            
            for i in range(self.max_neurons):
                if self.tier_map[i] == 0:  # T1 neuron
                    if grads[i] > self.grad_threshold_t2:
                        self.grad_counter[i] += 1
                        if self.grad_counter[i] >= self.patience:
                            self._upgrade_neuron(i)
                            self.grad_counter[i] = 0
                    else:
                        self.grad_counter[i] = max(0, self.grad_counter[i] - 1)
        
        # T2 → T3 growth
        for i, layer in enumerate(self.t2_layers):
            if layer.linear.weight.grad is not None:
                grad_norm = layer.linear.weight.grad.abs().mean().item()
                neuron_idx = self._t2_neuron_idx[i]  # melyik neuron ez
                if grad_norm > self.grad_threshold_t3:
                    self.t3_grad_counter[neuron_idx] += 1
                    if self.t3_grad_counter[neuron_idx] >= self.patience:
                        self._upgrade_neuron(neuron_idx)
                        self.t3_grad_counter[neuron_idx] = 0
                else:
                    self.t3_grad_counter[neuron_idx] = max(0, self.t3_grad_counter[neuron_idx] - 1)
    
    def get_stats(self):
        """Visszaadja a réteg statisztikáit."""
        n_t1 = (self.tier_map == 0).sum().item()
        n_t2 = (self.tier_map == 1).sum().item()
        n_t3 = (self.tier_map == 2).sum().item()
        return {
            'T1': n_t1, 'T2': n_t2, 'T3': n_t3,
            'total': n_t1 + n_t2 + n_t3,
            'step': self._step
        }


def test_growing():
    """
    Teszt: XOR feladat, growing mechanism-el.
    A neuronok T1-ként indulnak és figyeljük hogy nőnek-e.
    """
    torch.manual_seed(42)
    
    X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
    y = torch.tensor([[0.],[1.],[1.],[0.]])
    
    # Growing réteg
    layer = GrowingLayer(in_features=2, max_neurons=8,
                         grad_threshold_t2=0.005, patience=100)
    
    # Kimeneti réteg
    output = nn.Linear(8, 1)
    optimizer = torch.optim.Adam(list(layer.parameters()) + list(output.parameters()), lr=0.01)
    
    print("=== Growing Mechanism Teszt ===")
    print(f"Kezdő állapot: {layer.get_stats()}")
    
    for step in range(2000):
        # Forward
        h = layer(X, track_grads=True)
        pred = output(h)
        loss = F.mse_loss(pred, y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Growing + optimizer step
        layer.update_growth()
        optimizer.step()
        
        if step % 200 == 0:
            stats = layer.get_stats()
            print(f"  Step {step:4d}: loss={loss.item():.6f} | {stats}")
    
    print(f"\nVégső állapot: {layer.get_stats()}")
    print("Kimenetek:")
    with torch.no_grad():
        preds = output(layer(X))
        for inp, p, t in zip(X, preds, y):
            print(f"  {inp.tolist()} → {p.item():.3f} (várt: {t.item()})")


if __name__ == '__main__':
    test_growing()
