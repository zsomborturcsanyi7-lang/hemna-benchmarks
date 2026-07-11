"""Részletes debug: HEMNA v3 forward értékek"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, '.')
from hemna_v3 import HEMNAv3, generate_parity

torch.manual_seed(42)
X, y = generate_parity(4)

model = HEMNAv3(input_dim=4, hidden_sizes=[8], output_dim=1,
                grad_threshold_t2=0.001, grad_threshold_t3=0.002,
                patience=50)
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

print("=== RÉSZLETES DEBUG ===")
for step in range(300):
    pred = model(X)
    loss = F.mse_loss(pred, y)
    
    optimizer.zero_grad()
    loss.backward()
    model.update_growth()
    optimizer.step()
    
    if step in [0, 50, 100, 200]:
        print(f"\n--- Step {step} ---")
        stats = model.get_all_stats()[0]
        print(f"Stats: {stats}")
        
        with torch.no_grad():
            h = X
            for layer in model.layers:
                h = layer(h)
            
            base_dim = model.output_base.in_features
            in_dim = h.shape[-1]
            print(f"  Layer output shape: {h.shape}")
            print(f"  T1 part (first 8): min={h[:, :8].min().item():.4f}, max={h[:, :8].max().item():.4f}")
            if in_dim > 8:
                print(f"  T2 part (8:{in_dim}): min={h[:, 8:].min().item():.4f}, max={h[:, 8:].max().item():.4f}")
            
            out = model.output_base(h[..., :base_dim])
            print(f"  output_base(T1) min={out.min().item():.4f}, max={out.max().item():.4f}")
            
            if model.output_extra:
                extra_start = base_dim
                for j, w in enumerate(model.output_extra):
                    w_dim = w.shape[1]
                    if extra_start + w_dim <= in_dim:
                        extra = h[..., extra_start:extra_start + w_dim]
                        contrib = F.linear(extra, w, None)
                        out = out + contrib
                        extra_start += w_dim
                print(f"  output_extra count: {len(model.output_extra)}")
            
            print(f"  Final pred: min={out.min().item():.4f}, max={out.max().item():.4f}")
            print(f"  Sigmoid: {torch.sigmoid(out).squeeze()[:4].tolist()}")
            
            # T2 gradients
            layer = model.layers[0]
            t2_grads = []
            for i, l in enumerate(layer.t2_layers):
                if layer.tier_map[layer._t2_neuron_idx[i]] == 1:
                    gn = l.weight.grad.abs().mean().item() if l.weight.grad is not None else -1
                    t2_grads.append(gn)
            print(f"  T2 grads: {[f'{g:.4f}' for g in t2_grads[:5]]}{'...' if len(t2_grads) > 5 else ''}")

print("\n=== VÉG ===")
with torch.no_grad():
    preds = model(X)
    acc = ((preds > 0.5).float().eq(y).sum().item() / len(y) * 100)
print(f"Accuracy: {acc:.0f}%")
print(f"Stats: {model.get_all_stats()}")
