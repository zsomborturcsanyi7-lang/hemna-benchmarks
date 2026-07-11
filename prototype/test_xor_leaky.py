"""Teszt: XOR LeakyReLU-val"""
import torch, sys
sys.path.insert(0, '.')
from hemna_v3 import HEMNAv3

X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
y = torch.tensor([[0.],[1.],[1.],[0.]])

model = HEMNAv3(input_dim=2, hidden_sizes=[6], output_dim=1,
                grad_threshold_t2=0.001, grad_threshold_t3=0.005, patience=50)
opt = torch.optim.Adam(model.parameters(), lr=0.01)

print('Kezdo:', model.get_all_stats())
for step in range(1000):
    loss = model.train_step(X, y, opt)
    if step % 200 == 0:
        print(f'  Step {step}: loss={loss:.6f} | {model.get_all_stats()[0]}')

s = model.get_all_stats()[0]
print(f'Vegso: {s}')
with torch.no_grad():
    preds = model(X)
    for inp, p, t in zip(X, preds, y):
        print(f'  {inp.tolist()} -> {p.item():.4f} (vart: {t.item()})')
