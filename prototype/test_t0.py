import torch, sys
sys.path.insert(0, '.')
from hemna_v3 import HEMNAv3

X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
y = torch.tensor([[0.],[1.],[1.],[0.]])

# Alacsony threshold, hama nőjenek
m = HEMNAv3(2, [6], 1, 0.0001, 0.001, 0.002, 50)
o = torch.optim.Adam(m.parameters(), lr=0.01)

print('Kezdo:', m.get_all_stats())
for s in range(2000):
    loss = m.train_step(X, y, o)
    if s % 500 == 0:
        print(f'  Step {s}: loss={loss:.6f} | {m.get_all_stats()[0]}')

print('Vegso:', m.get_all_stats()[0])
with torch.no_grad():
    for i, p in zip(X, m(X)):
        print(f'  {i.tolist()} -> {p.item():.4f}')
