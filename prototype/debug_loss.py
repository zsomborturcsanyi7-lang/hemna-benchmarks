"""Loss debug + hasznalhato modell"""
import torch, torch.nn as nn, torch.nn.functional as F, sys, time, math
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(42)
text = """Az ember agya egy csodalatos szerkezet. Kutatasa mind a mai napig tart.
A mesterseges intelligencia egy olyan terulet ahol a szamitogepeket probaljuk 
emberi intelligenciaval felruhazni. A gepi tanulas segitsegevel a szamitogepek 
kepesek mintakat felismerni az adatokbol. A melytanulas tovabb lep tobb retegu 
halozatokkal dolgozik. A nyelvfeldolgozas az egyik legnehezebb feladat.
A mai idojaras napos es meleg. Holnap eso is lehet.
A macska az asztal alatt alszik. A kutya a kertben jatszik.
A matematika hatt feladatot kaptam. A programozas izgalmas.
Az alma egeszseges. A sport segit egeszsegesnek maradni.
A baratok tamaszt nyujtanak. Az utazas szinesiti az eletet.""".lower() * 50

chars = sorted(list(set(text)))
V = len(chars)
c2i = {c:i for i,c in enumerate(chars)}
i2c = {i:c for i,c in enumerate(chars)}
data = torch.tensor([c2i[c] for c in text])
n = len(data)
train_data, val_data = data[:int(n*0.9)], data[int(n*0.9):]
print(f"Karakterek: {len(text)}, szokincs: {V}")

def get_batch(data, bs=64, bl=64):
    ix = torch.randint(len(data)-bl-1, (bs,))
    x = torch.stack([data[i:i+bl] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+bl+1] for i in ix]).to(device)
    return x, y

# Egyszeru modell a gyors teszthez
class MiniModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(V, 64)
        self.attn = nn.MultiheadAttention(64, 4, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
        self.ln1, self.ln2 = nn.LayerNorm(64), nn.LayerNorm(64)
        self.out = nn.Linear(64, V)
    def forward(self, x):
        x = self.tok(x)
        # Causal mask
        mask = torch.triu(torch.ones(x.shape[1], x.shape[1], device=device) * float('-inf'), diagonal=1)
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ffn(x))
        return self.out(x)

print("\nLoss debug - tanulas elott:")
m = MiniModel().to(device)
x, y = get_batch(train_data, 32, 64)
logits = m(x)
loss = F.cross_entropy(logits.view(-1, V), y.view(-1))
print(f"  logits shape: {logits.shape}, range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
print(f"  loss (veletlen): {loss.item():.4f} (varhato: ~{math.log(V):.2f})")
assert abs(loss.item() - math.log(V)) < 0.5, f"Loss nem varhato! {loss.item()} vs {math.log(V)}"

print("\nTanitas 500 lepes...")
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
m.train()
for step in range(500):
    x, y = get_batch(train_data, 32, 64)
    opt.zero_grad()
    loss = F.cross_entropy(m(x).view(-1, V), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    if step % 100 == 0:
        m.eval()
        with torch.no_grad():
            xv, yv = get_batch(val_data, 32, 64)
            vl = F.cross_entropy(m(xv).view(-1, V), yv.view(-1)).item()
        print(f"  Step {step}: train_loss={loss.item():.4f}, val_loss={vl:.4f}")
        m.train()

print(f"\nVegso loss: {F.cross_entropy(m(get_batch(val_data,32,64)[0]).view(-1,V), get_batch(val_data,32,64)[1].view(-1)).item():.4f}")

print("\nGeneralas:")
with torch.no_grad():
    x = torch.tensor([[c2i['a']]], device=device)
    out = ['a']
    for _ in range(200):
        logits = m(x)
        probs = F.softmax(logits[0, -1] / 0.7, dim=-1)
        nxt = torch.multinomial(probs, 1).item()
        out.append(i2c[nxt])
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
        x = x[:, -64:]
    print(''.join(out))
