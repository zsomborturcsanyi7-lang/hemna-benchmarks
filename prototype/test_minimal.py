import torch, torch.nn as nn, torch.nn.functional as F, sys, time
device = 'cuda'
print(f"Minimal test on {device}")

V = 1000
m = nn.Sequential(nn.Embedding(V, 64), nn.Linear(64, V)).to(device)
x = torch.randint(0, V, (4, 128)).to(device)
y = torch.randint(0, V, (4, 128)).to(device)
opt = torch.optim.Adam(m.parameters(), lr=1e-3)

s = time.time()
for i in range(50):
    opt.zero_grad()
    l = F.cross_entropy(m(x).view(-1, V), y.view(-1))
    l.backward()
    opt.step()
    if i % 10 == 0:
        print(f"  Step {i}: loss={l.item():.4f}")
print(f"OK ({time.time()-s:.1f}s)")
