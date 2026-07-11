"""HEMNA v3 — optimizer újraépítés growth után"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, time
sys.path.insert(0, '.')
from hemna_v3 import HEMNAv3, generate_parity

torch.manual_seed(42)
X, y = generate_parity(4)

model = HEMNAv3(input_dim=4, hidden_sizes=[8], output_dim=1,
                grad_threshold_t2=0.001, grad_threshold_t3=0.002,
                patience=50)

# Optimizer rebuild helper
def make_opt():
    return torch.optim.Adam(model.parameters(), lr=0.02)

optimizer = make_opt()
last_n_params = len(list(model.parameters()))

print("=== 4-bit Parity — optimizer rebuild ===")
print(f"Kezdő: {model.get_all_stats()}")

start = time.time()
for step in range(3000):
    pred = model(X)
    loss = F.mse_loss(pred, y)
    
    optimizer.zero_grad()
    loss.backward()
    
    # Growth
    layer = model.layers[0]
    layer.update_growth()
    
    # Ha új paraméterek (nőtt a modell), új optimizer
    n_now = len(list(model.parameters()))
    if n_now > last_n_params:
        optimizer = make_opt()
        last_n_params = n_now
    
    # Adaptív output extra
    old_dim = model.layers[0].get_stats()['actual_dim']
    # (az update_growth már lefutott, actual_dim friss)
    # Nem kell külön output_extra mert model.parameters() már tartalmazza
    # a t2_layers és t3_layers új paramétereit
    
    optimizer.step()
    
    if step % 500 == 0:
        stats = model.get_all_stats()
        with torch.no_grad():
            preds = model(X)
            acc = ((preds > 0.5).float().eq(y).sum().item() / len(y) * 100)
        print(f"  Step {step:4d}: loss={loss.item():.6f} acc={acc:.0f}% | {stats[0]} | params={n_now}")

stats = model.get_all_stats()
print(f"Végső: {stats[0]}")

with torch.no_grad():
    preds = model(X)
    acc = ((preds > 0.5).float().eq(y).sum().item() / len(y) * 100)
print(f"Accuracy: {acc:.0f}%")
print(f"Idő: {time.time()-start:.1f}s")
