import torch, torch.nn as nn, torch.nn.functional as F, sys, time
from torchvision import datasets, transforms
sys.path.insert(0, '.')
from hemna_v3 import HEMNAv3, Tier0Linear, Tier1Linear, VectorizedBSplineLayer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Csak par batch a gyors teszthez
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,)), transforms.Lambda(lambda x: x.view(-1))])
train_data = datasets.MNIST('./data', train=True, download=False, transform=transform)
test_data = datasets.MNIST('./data', train=False, download=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=512)

# Csak 1 batch-en tesztelunk h lassu-e
print("Betoltes kesz, modell letrehozasa...")
s = time.time()
m = HEMNAv3(784, [32], 10, 0.001, 0.001, 0.002, 100).to(device)
print(f"Model created: {time.time()-s:.1f}s")
p = sum(p.numel() for p in m.parameters())
print(f"Params: {p}")

opt = torch.optim.Adam(m.parameters(), lr=0.001)
print("Elso forward+backward...")
s = time.time()
for d,t in train_loader:
    d,t = d.to(device), t.to(device)
    opt.zero_grad()
    out = m(d)
    loss = F.cross_entropy(out, t)
    loss.backward()
    m.update_growth()
    opt.step()
    stats = m.get_all_stats()[0]
    print(f"  1 batch: loss={loss.item():.4f} | {stats} ({time.time()-s:.1f}s)")
    break

# 1 teljes epoch
print(f"\n1 epoch training...")
s = time.time()
m.train()
for batch_idx, (d,t) in enumerate(train_loader):
    d,t = d.to(device), t.to(device)
    opt.zero_grad()
    loss = F.cross_entropy(m(d), t)
    loss.backward()
    m.update_growth()
    opt.step()
    if batch_idx % 100 == 0:
        stats = m.get_all_stats()[0]
        print(f"  batch {batch_idx}: loss={loss.item():.4f} | {stats}")
print(f"1 epoch kesz: {time.time()-s:.1f}s")
stats = m.get_all_stats()[0]
print(f"Vegso stat: {stats}")
