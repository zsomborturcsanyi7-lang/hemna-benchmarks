"""Parameter efficiency: MLP vs HEMNA - hany parameter kell ugyanahhoz az accuracy-hoz?"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import sys, time
sys.path.insert(0, '.')
from hemna_v3 import Tier1Linear

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: x.view(-1))
])
train = datasets.MNIST('./data', train=True, download=False, transform=transform)
test = datasets.MNIST('./data', train=False, download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=512)

def evaluate(model, loader):
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            if isinstance(model, torch.nn.Module):
                model.eval()
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    return correct / total

def count_params(*modules):
    return sum(p.numel() for m in modules for p in m.parameters())

epochs = 3

print(f"\n{'Modell':<25} {'Neuronok':<15} {'Params':<10} {'Epoch1':<10} {'Epoch3':<10} {'Test':<10}")
print("-" * 80)

results = []

# === MLP szériák ===
for n in [16, 32, 64, 128]:
    mlp = nn.Sequential(
        nn.Linear(784, n), nn.ReLU(),
        nn.Linear(n, 10)
    ).to(device)
    p = count_params(mlp)
    opt = torch.optim.Adam(mlp.parameters(), lr=0.001)
    
    for ep in range(epochs):
        mlp.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(mlp(data), target)
            loss.backward()
            opt.step()
        acc = evaluate(mlp, test_loader)
        if ep == epochs - 1:
            print(f"{'MLP ReLU':<25} {f'{n}':<15} {p:<10} {'':<10} {'':<10} {acc:<8.2%}")
            results.append(('MLP', n, p, acc))

# === HEMNA T1+T2 szériák ===
for total_n in [16, 32, 64, 128]:
    t1_n = total_n // 2
    t2_n = total_n - t1_n
    t1 = Tier1Linear(784, t1_n).to(device)
    t2 = nn.Linear(784, t2_n).to(device)
    out = nn.Linear(total_n, 10).to(device)
    p = count_params(t1, t2, out)
    opt = torch.optim.Adam(list(t1.parameters())+list(t2.parameters())+list(out.parameters()), lr=0.001)
    
    for ep in range(epochs):
        t1.train(); t2.train(); out.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            opt.zero_grad()
            h = torch.cat([t1(data), F.relu(t2(data))], dim=-1)
            loss = F.cross_entropy(out(h), target)
            loss.backward()
            opt.step()
        acc = evaluate(lambda x: out(torch.cat([t1(x), F.relu(t2(x))], -1)), test_loader)
        if ep == epochs - 1:
            print(f"{'HEMNA T1+T2':<25} {f'{t1_n}+{t2_n}':<15} {p:<10} {'':<10} {'':<10} {acc:<8.2%}")
            results.append(('HEMNA', total_n, p, acc))

print("\n=== Parameter efficiency at 95% ===")
print(f"{'Modell':<20} {'Params@95%':<15}")
# Find smallest model that hits 95%
for name in ['MLP', 'HEMNA']:
    for model_name, n, p, acc in results:
        if model_name == name and acc >= 0.95:
            print(f"  {name:<20} {p:<15} ({n} neurons, {acc:.2%})")
            break

print("\n=== Parameter efficiency at 97% ===")
for name in ['MLP', 'HEMNA']:
    for model_name, n, p, acc in results:
        if model_name == name and acc >= 0.97:
            print(f"  {name:<20} {p:<15} ({n} neurons, {acc:.2%})")
            break
