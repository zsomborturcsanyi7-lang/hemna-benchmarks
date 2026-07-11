"""MNIST benchmark: Standard MLP vs HEMNA v3"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import sys, time
sys.path.insert(0, '.')
from hemna_v3 import Tier1Linear, VectorizedBSplineLayer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: x.view(-1))
])
train = datasets.MNIST('./data', train=True, download=True, transform=transform)
test = datasets.MNIST('./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=512)

def train_epoch(model, loader, opt):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        opt.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        opt.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    return total_loss / len(loader), correct / total

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    return correct / total

epochs = 5

# === Standard MLP ===
print("\n=== MLP (256 ReLU) ===")
torch.manual_seed(42)
mlp = nn.Sequential(
    nn.Linear(784, 256), nn.ReLU(),
    nn.Linear(256, 10)
).to(device)
opt = torch.optim.Adam(mlp.parameters(), lr=0.001)
p = sum(p.numel() for p in mlp.parameters())
print(f"  Parameters: {p}")
for ep in range(epochs):
    s = time.time()
    loss, acc = train_epoch(mlp, train_loader, opt)
    test_acc = evaluate(mlp, test_loader)
    print(f"  Epoch {ep+1}: train_loss={loss:.4f} train_acc={acc:.2%} test_acc={test_acc:.2%} ({time.time()-s:.1f}s)")

# === HEMNA T1+T2 (no T3) ===
print("\n=== HEMNA (128 T1 + 128 T2) ===")
torch.manual_seed(42)
t1 = Tier1Linear(784, 128).to(device)
t2 = nn.Linear(784, 128).to(device)
out = nn.Linear(256, 10).to(device)
opt2 = torch.optim.Adam(list(t1.parameters())+list(t2.parameters())+list(out.parameters()), lr=0.001)
p2 = sum(p.numel() for p in [*t1.parameters(), *t2.parameters(), *out.parameters()])
print(f"  Parameters: {p2}")
for ep in range(epochs):
    s = time.time()
    def forward_t1t2(x):
        h = torch.cat([t1(x), F.relu(t2(x))], dim=-1)
        return out(h)
    model_t1t2 = lambda x: forward_t1t2(x)
    # train manually
    t1.train(); t2.train(); out.train()
    total_loss, correct, total = 0, 0, 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        opt2.zero_grad()
        h = torch.cat([t1(data), F.relu(t2(data))], dim=-1)
        output = out(h)
        loss = F.cross_entropy(output, target)
        loss.backward()
        opt2.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    train_acc = correct / total
    # test
    t1.eval(); t2.eval(); out.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            h = torch.cat([t1(data), F.relu(t2(data))], dim=-1)
            output = out(h)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    test_acc = correct / total
    print(f"  Epoch {ep+1}: train_loss={total_loss/len(train_loader):.4f} train_acc={train_acc:.2%} test_acc={test_acc:.2%} ({time.time()-s:.1f}s)")

# === HEMNA T1+T2+T3 ===
print("\n=== HEMNA (64 T1 + 64 T2 + 64 T3) ===")
torch.manual_seed(42)
t1b = Tier1Linear(784, 64).to(device)
t2b = nn.Linear(784, 64).to(device)
t3b = VectorizedBSplineLayer(784, 64).to(device)  # 784in -> 64out, grid=-2..2
outb = nn.Linear(192, 10).to(device)
opt3 = torch.optim.Adam(
    list(t1b.parameters())+list(t2b.parameters())+list(t3b.parameters())+list(outb.parameters()),
    lr=0.001)
p3 = sum(p.numel() for p in [*t1b.parameters(), *t2b.parameters(), *t3b.parameters(), *outb.parameters()])
print(f"  Parameters: {p3}")
for ep in range(epochs):
    s = time.time()
    t1b.train(); t2b.train(); t3b.train(); outb.train()
    total_loss, correct, total = 0, 0, 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        opt3.zero_grad()
        h = torch.cat([t1b(data), F.relu(t2b(data)), t3b(data)], dim=-1)
        output = outb(h)
        loss = F.cross_entropy(output, target)
        loss.backward()
        opt3.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    train_acc = correct / total
    t1b.eval(); t2b.eval(); t3b.eval(); outb.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            h = torch.cat([t1b(data), F.relu(t2b(data)), t3b(data)], dim=-1)
            output = outb(h)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    test_acc = correct / total
    print(f"  Epoch {ep+1}: train_loss={total_loss/len(train_loader):.4f} train_acc={train_acc:.2%} test_acc={test_acc:.2%} ({time.time()-s:.1f}s)")

print(f"\n=== OSSZEGZES ===")
print(f"MLP ({p} param):              test_acc={evaluate(mlp, test_loader):.2%}")
print(f"HEMNA T1+T2 ({p2} param):     test_acc={evaluate(lambda x: out(torch.cat([t1(x),F.relu(t2(x))],-1)), test_loader):.2%}")
print(f"HEMNA T1+T2+T3 ({p3} param):  test_acc={evaluate(lambda x: outb(torch.cat([t1b(x),F.relu(t2b(x)),t3b(x)],-1)), test_loader):.2%}")
