"""MNIST benchmark T0->T1->T2->T3 architekturaval"""
import torch, torch.nn as nn, torch.nn.functional as F, sys, time
from torchvision import datasets, transforms
sys.path.insert(0, '.')
from hemna_v3 import HEMNAv3

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,)), transforms.Lambda(lambda x: x.view(-1))])
train = datasets.MNIST('./data', train=True, download=True, transform=transform)
test = datasets.MNIST('./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=512)

epochs = 5

# Standard MLP
print("\n=== MLP 256 ReLU ===")
mlp = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10)).to(device)
opt = torch.optim.Adam(mlp.parameters(), lr=0.001)
p = sum(p.numel() for p in mlp.parameters())
print(f"Params: {p}")
for ep in range(epochs):
    s = time.time(); mlp.train()
    for d, t in train_loader:
        d, t = d.to(device), t.to(device)
        opt.zero_grad(); loss = F.cross_entropy(mlp(d), t); loss.backward(); opt.step()
    mlp.eval()
    with torch.no_grad():
        correct = sum((mlp(d.to(device)).argmax(1) == t.to(device)).sum().item() for d, t in test_loader)
        print(f"  Epoch {ep+1}: test_acc={correct/len(test):.2%} ({time.time()-s:.1f}s)")

# HEMNA T0->T1->T2->T3
print("\n=== HEMNA T0->T1->T2 (64 neuron) ===")
m = HEMNAv3(784, [64], 10, 0.001, 0.001, 100).to(device)
opt2 = torch.optim.Adam(m.parameters(), lr=0.001)
p2 = sum(p.numel() for p in m.parameters())
print(f"Params: {p2}")
for ep in range(epochs):
    s = time.time(); m.train()
    for d, t in train_loader:
        d, t = d.to(device), t.to(device)
        opt2.zero_grad(); loss = F.cross_entropy(m(d), t); loss.backward(); opt2.step()
        m.update_growth()
    m.eval()
    with torch.no_grad():
        correct = sum((m(d.to(device)).argmax(1) == t.to(device)).sum().item() for d, t in test_loader)
        stats = m.get_all_stats()[0]
        print(f"  Epoch {ep+1}: test_acc={correct/len(test):.2%} {stats} ({time.time()-s:.1f}s)")

print(f"\nMLP:  {p} params")
print(f"HEMNA: {p2} params")
