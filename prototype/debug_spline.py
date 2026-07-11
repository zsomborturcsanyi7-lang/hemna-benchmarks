"""Debug: mit ad ki a BSpline és a Tier3Linear?"""
import torch
import sys
sys.path.insert(0, '.')
from hemna_v3 import FastBSpline, Tier3Linear, HEMNAv3

# 1. BSpline debug
print("=== BSpline debug ===")
spline = FastBSpline(grid_size=8, degree=3)
print(f"coeffs: {spline.coefficients.data}")
print(f"grid: {spline.grid}")

for x in [0.0, 0.5, 1.0]:
    t = torch.tensor([x])
    out = spline(t)
    print(f"  BSpline({x}) = {out.item():.6f}")

# 2. Tier3Linear debug
print("\n=== Tier3Linear debug ===")
torch.manual_seed(42)
t3 = Tier3Linear(2, 1)
x = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
out = t3(x)
print(f"output: {out.squeeze().tolist()}")
print(f"mean: {out.mean().item():.4f}, std: {out.std().item():.4f}")
print(f"sigmoid: {torch.sigmoid(out).squeeze().tolist()}")

# 3. Teljes modell debug
print("\n=== HEMNAv3 debug (XOR) ===")
torch.manual_seed(42)
model = HEMNAv3(input_dim=2, hidden_sizes=[6], output_dim=1,
                grad_threshold_t2=0.001, grad_threshold_t3=0.005,
                patience=50)
X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
y = torch.tensor([[0.],[1.],[1.],[0.]])

# Forward before any training
with torch.no_grad():
    pred = model(X)
    print(f"Before training - pred: {pred.squeeze().tolist()}")
    print(f"Before training - loss: {torch.nn.functional.mse_loss(pred, y).item():.6f}")

# One training step
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss = model.train_step(X, y, optimizer)
print(f"After step 1 - loss: {loss:.6f}")

# Check if gradients flow
with torch.no_grad():
    pred = model(X)
    print(f"After step 1 - pred: {pred.squeeze().tolist()}")
    
# Manually compute what the model outputs
print("\n=== Layer-by-layer debug ===")
with torch.no_grad():
    h = X
    for i, layer in enumerate(model.layers):
        h = layer(h)
        print(f"  Layer {i} output shape: {h.shape}")
        print(f"  Layer {i} output range: [{h.min().item():.4f}, {h.max().item():.4f}]")
    
    # Output
    base_dim = model.output_base.in_features
    in_dim = h.shape[-1]
    print(f"  Output base dim: {base_dim}, actual dim: {in_dim}")
    
    if in_dim <= base_dim:
        out = model.output_base(h)
    else:
        out = model.output_base(h[..., :base_dim])
        extra_start = base_dim
        for j, w in enumerate(model.output_extra):
            w_dim = w.shape[1]
            if extra_start + w_dim <= in_dim:
                extra = h[..., extra_start:extra_start + w_dim]
                out = out + torch.nn.functional.linear(extra, w, None)
                print(f"  Extra weight {j}: shape={w.shape}, range=[{w.min().item():.4f}, {w.max().item():.4f}]")
                print(f"  Extra input range: [{extra.min().item():.4f}, {extra.max().item():.4f}]")
                extra_start += w_dim
    
    print(f"  Output: {out.squeeze().tolist()}")
