import torch, torch.nn as nn, torch.nn.functional as F, sys, time
from torchvision import datasets, transforms
sys.path.insert(0, '.')
from hemna_v3 import HEMNAv3

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,)), transforms.Lambda(lambda x: x.view(-1))])
train = datasets.MNIST('./data', train=True, download=False, transform=transform)
test = datasets.MNIST('./data', train=False, download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=512)

print("Loading MLP...")
mlp = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10)).to(device)
opt = torch.optim.Adam(mlp.parameters(), lr=0.001)

print("Training MLP 5 epochs...")
for ep in range(5):
    s = time.time(); mlp.train()
    for d, t in train_loader:
        d, t = d.to(device), t.to(device)
        opt.zero_grad(); F.cross_entropy(mlp(d), t).backward(); opt.step()
    mlp.eval()
    with torch.no_grad():
        correct = sum((mlp(d.to(device)).argmax(1) == t.to(device)).sum().item() for d, t in test_loader)
    print(f"  Epoch {ep+1}: {correct/len(test):.2%} ({time.time()-s:.1f}s)")

print(f"\nMLP done. Creating HEMNA...")
s = time.time()
m = HEMNAv3(784, [64], 10, 0.001, 0.001, 100).to(device)
print(f"HEMNA created: {time.time()-s:.1f}s, params={sum(p.numel() for p in m.parameters())}")
opt2 = torch.optim.Adam(m.parameters(), lr=0.001)

print("HEMNA 1 epoch...")
s = time.time(); m.train()
for batch_idx, (d, t) in enumerate(train_loader):
    d, t = d.to(device), t.to(device)
    opt2.zero_grad()
    out = m(d)
    loss = F.cross_entropy(out, t)
    loss.backward()
    m.update_growth()
    opt2.step()
    if batch_idx % 200 == 0:
        print(f"  batch {batch_idx}: loss={loss.item():.4f} | {m.get_all_stats()[0]}")

m.eval()
with torch.no_grad():
    correct = sum((m(d.to(device)).argmax(1) == t.to(device)).sum().item() for d, t in test_loader)
print(f"  Test: {correct/len(test):.2%} ({time.time()-s:.1f}s)")
