"""
HEMNA v3 vs Standard MLP — MNIST benchmark

Összehasonlítás: ugyanannyi paraméterrel, fix architektúra.
HEMNA v3: 1 réteg, 64 neuron, tier arányokkal
Standard MLP: 1 réteg, 64 neuron, ReLU
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, time
sys.path.insert(0, '.')
from hemna_v3 import Tier1Linear, Tier2Linear, Tier3Linear, FastBSpline

# ============================================================
# HEMNA v3 — fix tier arány (nincs growth)
# ============================================================
class HEMNAv3Fixed(nn.Module):
    """HEMNA v3 fix tier arányokkal — nincs dinamikus growth."""
    def __init__(self, input_dim, hidden_dim, output_dim,
                 t1_ratio=0.25, t2_ratio=0.5, t3_ratio=0.25):
        super().__init__()
        n_t1 = max(1, int(hidden_dim * t1_ratio))
        n_t2 = max(1, int(hidden_dim * t2_ratio))
        n_t3 = hidden_dim - n_t1 - n_t2
        
        print(f"  HEMNA: {n_t1}T1 + {n_t2}T2 + {n_t3}T3 = {n_t1+n_t2+n_t3} neuron")
        
        self.t1 = Tier1Linear(input_dim, n_t1)
        self.t2 = Tier2Linear(input_dim, n_t2)
        self.t3 = Tier3Linear(input_dim, n_t3)
        self.output = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        t1_out = self.t1(x)
        t2_out = self.t2(x)
        t3_out = self.t3(x)
        h = torch.cat([t1_out, t2_out, t3_out], dim=-1)
        return self.output(h)


# ============================================================
# MNIST betöltés
# ============================================================
def load_mnist():
    """MNIST betöltés és előfeldolgozás."""
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    
    train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=512)
    
    return train_loader, test_loader


def train_epoch(model, loader, optimizer, device='cpu'):
    """Egy epoch tanítás."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    return total_loss / len(loader), correct / total


def evaluate(model, loader, device='cpu'):
    """Kiértékelés."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return correct / total


# ============================================================
# Benchmark
# ============================================================
print("=== MNIST Benchmark ===")
print()

# Adatok betöltése
print("MNIST betöltése...")
train_loader, test_loader = load_mnist()
print(f"  Train: {len(train_loader.dataset)} images")
print(f"  Test: {len(test_loader.dataset)} images")
print()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# ===== Standard MLP =====
print("\n--- Standard MLP (1 réteg, 64 ReLU) ---")
torch.manual_seed(42)
mlp = nn.Sequential(
    nn.Linear(784, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
).to(device)
opt_mlp = torch.optim.Adam(mlp.parameters(), lr=0.001)
mlp_params = sum(p.numel() for p in mlp.parameters())
print(f"  Paraméterek: {mlp_params}")

# ===== HEMNA v3 fixed =====
print("\n--- HEMNA v3 (25% T1 + 50% T2 + 25% T3) ---")
torch.manual_seed(42)
hemna = HEMNAv3Fixed(784, 64, 10, t1_ratio=0.25, t2_ratio=0.5, t3_ratio=0.25).to(device)
opt_hemna = torch.optim.Adam(hemna.parameters(), lr=0.001)
hemna_params = sum(p.numel() for p in hemna.parameters())
print(f"  Paraméterek: {hemna_params}")

# ===== HEMNA v3 more T1 =====
print("\n--- HEMNA v3 (50% T1 + 30% T2 + 20% T3) ---")
torch.manual_seed(42)
hemna2 = HEMNAv3Fixed(784, 64, 10, t1_ratio=0.5, t2_ratio=0.3, t3_ratio=0.2).to(device)
opt_hemna2 = torch.optim.Adam(hemna2.parameters(), lr=0.001)

# Training loop
epochs = 5
models = [
    ("MLP", mlp, opt_mlp),
    ("HEMNA (25/50/25)", hemna, opt_hemna),
    ("HEMNA (50/30/20)", hemna2, opt_hemna2),
]

print(f"\n{'Modell':<20} {'Epoch':>6} {'Train Loss':>12} {'Train Acc':>10} {'Test Acc':>9}")
print("-" * 60)

for name, model, opt in models:
    for epoch in range(epochs):
        start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, opt, device)
        test_acc = evaluate(model, test_loader, device)
        t = time.time() - start
        print(f"{name:<20} {epoch+1:>4}/{epochs} {train_loss:>10.4f} {train_acc:>8.2%} {test_acc:>7.2%}  ({t:.1f}s)")
    print()
