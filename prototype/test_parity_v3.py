import torch, sys
sys.path.insert(0, '.')
from hemna_v3 import HEMNAv3, generate_parity

torch.manual_seed(42)
X, y = generate_parity(4)

model = HEMNAv3(4, [8], 1, 0.001, 0.002, 50)
opt = torch.optim.Adam(model.parameters(), lr=0.01)
print('Kezdo:', model.get_all_stats())

for s in range(3000):
    loss = model.train_step(X, y, opt)
    if s % 500 == 0:
        print(f'  Step {s}: loss={loss:.6f} | {model.get_all_stats()[0]}')

print()
with torch.no_grad():
    preds = model(X)
    acc = ((preds > 0.5).float().eq(y).sum().item() / len(y) * 100)
    print(f'Accuracy: {acc:.0f}%')
print('Vegso:', model.get_all_stats()[0])
