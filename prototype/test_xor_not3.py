import torch, sys
sys.path.insert(0, '.')
from hemna_v3 import HEMNAv3

X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
y = torch.tensor([[0.],[1.],[1.],[0.]])

# T3 threshold olyan magas hogy sosem aktivallodjon
model = HEMNAv3(2, [6], 1, 0.001, 999.0, 50)
opt = torch.optim.Adam(model.parameters(), lr=0.01)
print('Kezdo:', model.get_all_stats())

for s in range(2000):
    loss = model.train_step(X, y, opt)
    if s % 500 == 0:
        stats = model.get_all_stats()[0]
        print(f'  Step {s}: loss={loss:.6f} | T1={stats["T1"]} T2={stats["T2"]} T3={stats["T3"]}')

print()
with torch.no_grad():
    for i, p, t in zip(X, model(X), y):
        print(f'  {i.tolist()} -> {p.item():.4f} (vart: {t.item()})')
