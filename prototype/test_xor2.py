import torch, sys
sys.path.insert(0, '.')
from hemna_v3 import HEMNAv3

X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
y = torch.tensor([[0.],[1.],[1.],[0.]])

model = HEMNAv3(2, [6], 1, 0.001, 0.005, 50)
opt = torch.optim.Adam(model.parameters(), lr=0.01)
print(model.get_all_stats())

for s in range(1000):
    loss = model.train_step(X, y, opt)
    if s % 200 == 0:
        print(f'  Step {s}: loss={loss:.6f} | {model.get_all_stats()[0]}')

print('Vegso:', model.get_all_stats()[0])
with torch.no_grad():
    for i, p, t in zip(X, model(X), y):
        print(f'  {i.tolist()} -> {p.item():.4f} (vart: {t.item()})')
