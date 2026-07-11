"""HEMNA v3 parity hosszabb teszt"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, time
sys.path.insert(0, '.')
from hemna_v3 import HEMNAv3, generate_parity

# === 4-bit Parity ===
torch.manual_seed(42)
X, y = generate_parity(4)

model = HEMNAv3(input_dim=4, hidden_sizes=[8], output_dim=1,
                grad_threshold_t2=0.001, grad_threshold_t3=0.002,
                patience=50)
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)  # higher LR

print("=== 4-bit Parity — hosszú teszt ===")
print(f"Kezdő: {model.get_all_stats()}")

start = time.time()
for step in range(5000):
    loss = model.train_step(X, y, optimizer)
    
    if step % 1000 == 0:
        stats = model.get_all_stats()
        with torch.no_grad():
            preds = model(X)
            acc = ((preds > 0.5).float().eq(y).sum().item() / len(y) * 100)
        print(f"  Step {step:4d}: loss={loss:.6f} acc={acc:.0f}% | {stats[0]}")

stats = model.get_all_stats()
print(f"Végső: {stats[0]}")

with torch.no_grad():
    preds = model(X)
    acc = ((preds > 0.5).float().eq(y).sum().item() / len(y) * 100)
print(f"Accuracy: {acc:.0f}%")
print(f"Idő: {time.time()-start:.1f}s")

# === Standard MLP benchmark ===
print("\n=== Standard MLP benchmark (2 réteg, ReLU) ===")
torch.manual_seed(42)
mlp = nn.Sequential(
    nn.Linear(4, 8), nn.ReLU(),
    nn.Linear(8, 1)
)
opt2 = torch.optim.Adam(mlp.parameters(), lr=0.02)

for step in range(5000):
    pred = mlp(X)
    loss = F.mse_loss(pred, y)
    opt2.zero_grad()
    loss.backward()
    opt2.step()
    
    if step % 1000 == 0:
        with torch.no_grad():
            acc = ((pred > 0.5).float().eq(y).sum().item() / len(y) * 100)
        print(f"  Step {step:4d}: loss={loss.item():.6f} acc={acc:.0f}%")

with torch.no_grad():
    preds = mlp(X)
    acc = ((preds > 0.5).float().eq(y).sum().item() / len(y) * 100)
print(f"MLP Accuracy: {acc:.0f}%")
