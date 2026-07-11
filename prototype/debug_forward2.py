"""Debug: forward értékek a pre-allocated verzióban"""
import torch, sys
sys.path.insert(0, '.')
from hemna_v3 import HEMNAv3

torch.manual_seed(42)
X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
y = torch.tensor([[0.],[1.],[1.],[0.]])

model = HEMNAv3(input_dim=2, hidden_sizes=[6], output_dim=1,
                grad_threshold_t2=0.001, grad_threshold_t3=0.005,
                patience=50)

# Forward before training
with torch.no_grad():
    h = X
    for layer in model.layers:
        h = layer(h)
        print(f"  Layer output: min={h.min().item():.4f}, max={h.max().item():.4f}")
        print(f"  Non-zero count: {(h.abs() > 1e-6).sum().item()} / {h.numel()}")
    
    pred = model.output(h)
    print(f"  Pred before training: {pred.tolist()}")
    print(f"  Output weight: {model.output.weight.tolist()}")
    print(f"  Output bias: {model.output.bias.tolist()}")

# Training 5 steps manually
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for step in range(5):
    loss = model.train_step(X, y, optimizer)
    with torch.no_grad():
        h = X
        for layer in model.layers:
            h = layer(h)
        pred = model.output(h)
    print(f"\nStep {step}: loss={loss:.6f}, pred={pred.squeeze().tolist()}, "
          f"h range=[{h.min().item():.4f}, {h.max().item():.4f}]")
    stats = model.get_all_stats()[0]
    print(f"  Stats: {stats}")
    # Check for NaN
    for name, p in model.named_parameters():
        if torch.isnan(p).any():
            print(f"  NAN in {name}!")
