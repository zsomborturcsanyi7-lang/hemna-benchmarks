"""Debug: gradient flow check XOR-on"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, '.')
from hemna_v3 import HEMNAv3

torch.manual_seed(42)
X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
y = torch.tensor([[0.],[1.],[1.],[0.]])

model = HEMNAv3(input_dim=2, hidden_sizes=[6], output_dim=1,
                grad_threshold_t2=0.001, grad_threshold_t3=0.005,
                patience=50)

# Manuális training loop gradiens nyomkövetéssel
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for step in range(200):
    pred = model(X)
    loss = F.mse_loss(pred, y)
    
    optimizer.zero_grad()
    loss.backward()
    
    # Nyomkövetés
    if step % 50 == 0 or step < 5:
        layer = model.layers[0]
        stats = layer.get_stats()
        
        # T2 gradiens norm
        t2_grads = []
        for i, l in enumerate(layer.t2_layers):
            if layer.tier_map[layer._t2_neuron_idx[i]] == 1:
                gn = l.weight.grad.abs().mean().item() if l.weight.grad is not None else -1
                t2_grads.append(gn)
        
        # T1 gradiens norm
        if layer.t1.weight.grad is not None:
            t1_grads = layer.t1.weight.grad.abs().mean(dim=1).tolist()
        else:
            t1_grads = [-1]*6
        
        print(f"Step {step:3d}: loss={loss.item():.4f} | {stats}")
        print(f"  T1 grads: {[f'{g:.4f}' for g in t1_grads]}")
        print(f"  T2 grads: {[f'{g:.4f}' for g in t2_grads]}")
        print(f"  T2→T3 counter: {layer.t3_grad_counter.tolist()}")
        print(f"  Pred: {pred.squeeze().tolist()}")
    
    new_params = model.update_growth()
    if new_params:
        optimizer.add_param_group({'params': new_params})
    optimizer.step()

print("\n=== VÉG ===")
stats = model.get_all_stats()
print(stats)
with torch.no_grad():
    preds = model(X)
    for inp, p, t in zip(X, preds, y):
        print(f"  {inp.tolist()} → {p.item():.4f} (várt: {t.item()})")
