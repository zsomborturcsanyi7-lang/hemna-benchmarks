"""Javitott mikro nyelvi modell - Sliding Window + GQA + Gated FFN + Dropout"""
import torch, torch.nn as nn, torch.nn.functional as F, sys, time, math
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# ============================================================
# 1. Adat
# ============================================================
torch.manual_seed(42)
base = """Az ember agya egy csodalatos szerkezet. Kutatasa mind a mai napig tart.
A mesterseges intelligencia egy olyan terulet ahol a szamitogepeket probaljuk 
emberi intelligenciaval felruhazni. A gepi tanulas segitsegevel a szamitogepek 
kepesek mintakat felismerni az adatokbol. A melytanulas tovabb lep tobb retegu 
halozatokkal dolgozik. A nyelvfeldolgozas az egyik legnehezebb feladat. 
Egy gepnek meg kell ertenie a szavak jelentese kozotti osszefuggeseket. 
A transformer architektura forradalmasitotta a termeszetes nyelvfeldolgozast. 
Az attention mechanizmus minden szot minden szoval osszekot. 
A jovo mesterseges inteligenciaja meg tobbet fog tudni.
A kutatok ujabb es ujabb modszereket fejlesztenek. 
A cel egy olyan gep letrehozasa ami tenylegesen megerti az emberi nyelvet."""

extra = [
    "A mai idojaras napos es meleg lesz. Holnap varnato eso is lehet.",
    "A macska az asztal alatt alszik. A kutya a kertben jatszik.",
    "A matek hatt feladatot kaptam. Meg kell oldanom a harmadik peldat.",
    "Az alma egeszseges gyumolcs. Naponta egy alma az orvost tavol tartja.",
    "A programozas izgalmas hobbi. Uj nyelveket tanulni mindig erdekes.",
    "A konyvek fontosak a tanulashoz. Egy jo konyv eletre szolo tudast ad.",
    "A sport segit egeszsegesnek maradni. Minden nap edzek egy kicsit.",
    "A baratok tamaszt nyujtanak a nehez idokben. Egy jo barat ritka kincs.",
    "Az utazas szinesiti az eletet. Uj helyeket felfedezni csodas dolog.",
    "A technologia folyamatosan fejlodik. Minden evben uj talalmanyok.",
]
corpus = '\n'.join([base] + extra) * 200

# Karakter szintu tokenizacio (egyszeru, gyors, megbizhato)
chars = sorted(list(set(corpus)))
V = len(chars)
c2i = {c:i for i,c in enumerate(chars)}
i2c = {i:c for i,c in enumerate(chars)}
data = torch.tensor([c2i[c] for c in corpus], dtype=torch.long)
n = len(data)
train_data, val_data = data[:int(n*0.9)], data[int(n*0.9):]
print(f"Korpusz: {len(corpus)} karakter, szokincs: {V}")

def get_batch(data, bs=32, bl=256):
    ix = torch.randint(len(data)-bl-1, (bs,))
    x = torch.stack([data[i:i+bl] for i in ix])
    y = torch.stack([data[i+1:i+bl+1] for i in ix])
    return x.to(device), y.to(device)

def count_params(m):
    return sum(p.numel() for p in m.parameters())

def generate(model, prompt="\n", max_len=200, temp=0.8):
    model.eval()
    with torch.no_grad():
        x = torch.tensor([c2i[c] for c in prompt], device=device).unsqueeze(0)
        out = list(prompt)
        for _ in range(max_len):
            logits = model(x[:, -256:])
            probs = F.softmax(logits[0, -1] / temp, dim=-1)
            next_c = torch.multinomial(probs, 1).item()
            out.append(i2c[next_c])
            x = torch.cat([x, torch.tensor([[next_c]], device=device)], dim=1)
        return ''.join(out)

# ============================================================
# 2. Architektura - javitasokkal
# ============================================================
class SwiGLU(nn.Module):
    def __init__(self, dim, h, drop=0.1):
        super().__init__()
        self.w1 = nn.Linear(dim, h, False)
        self.w2 = nn.Linear(h, dim, False)
        self.w3 = nn.Linear(dim, h, False)
        self.d = nn.Dropout(drop)
    def forward(self, x):
        return self.d(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class SWAttn(nn.Module):
    """Sliding Window + GQA."""
    def __init__(self, dim, nh, nkv, win=128, drop=0.1):
        super().__init__()
        self.nh, self.nkv, self.nr, self.hd = nh, nkv, nh//nkv, dim//nh
        self.win = win
        self.wq = nn.Linear(dim, nh*self.hd, False)
        self.wk = nn.Linear(dim, nkv*self.hd, False)
        self.wv = nn.Linear(dim, nkv*self.hd, False)
        self.wo = nn.Linear(nh*self.hd, dim, False)
        self.d = nn.Dropout(drop)
        self.register_buffer('cm', torch.tril(torch.ones(512, 512)))
    
    def forward(self, x):
        B,T,D = x.shape
        q = self.wq(x).view(B,T,self.nh,self.hd).transpose(1,2)
        k = self.wk(x).view(B,T,self.nkv,self.hd).transpose(1,2)
        v = self.wv(x).view(B,T,self.nkv,self.hd).transpose(1,2)
        if self.nr > 1:
            k = k[:,:,None].expand(-1,-1,self.nr,-1,-1).reshape(B,self.nh,T,self.hd)
            v = v[:,:,None].expand(-1,-1,self.nr,-1,-1).reshape(B,self.nh,T,self.hd)
        wei = (q @ k.transpose(-2,-1)) * (self.hd**-0.5)
        wei = wei.masked_fill(self.cm[:T,:T]==0, float('-inf'))
        wei = wei.masked_fill(torch.triu(torch.ones(T,T,device=x.device), self.win+1)==1, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        out = (self.d(wei) @ v).transpose(1, 2).reshape(B, T, -1)
        return self.wo(out)

class Block(nn.Module):
    def __init__(self, dim, nh, nkv, ffn, win=128, drop=0.1):
        super().__init__()
        self.sa = SWAttn(dim, nh, nkv, win, drop)
        self.ff = SwiGLU(dim, ffn, drop)
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.d = nn.Dropout(drop)
    def forward(self, x):
        x = x + self.d(self.sa(self.ln1(x)))
        x = x + self.d(self.ff(self.ln2(x)))
        return x

class MicroLM(nn.Module):
    def __init__(self, V, dim=256, nl=6, nh=8, ffn=512, win=128, drop=0.1):
        super().__init__()
        self.tok = nn.Embedding(V, dim)
        self.d = nn.Dropout(drop)
        self.blocks = nn.ModuleList([Block(dim, nh, max(2,nh//4), ffn, win, drop) for _ in range(nl)])
        self.ln = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, V, False)
    def forward(self, x):
        x = self.d(self.tok(x))
        for b in self.blocks:
            x = b(x)
        return self.out(self.ln(x))

# ============================================================
# 3. Benchmark
# ============================================================
confs = [
    ("Alap (128d-4l-256ffn-win64)", 128, 4, 4, 256, 64, 0.1),
    ("Javitott (256d-6l-512ffn-win128)", 256, 6, 8, 512, 128, 0.1),
    ("Mely (128d-8l-256ffn-win64)", 128, 8, 4, 256, 64, 0.15),
]

print(f"\n{'Konfig':<40} {'Params':<10} {'Loss':<10} {'PPL':<10} {'Ido':<8}")
print("-" * 85)

for label, dim, nl, nh, ffn, win, drop in confs:
    torch.manual_seed(42)
    m = MicroLM(V, dim, nl, nh, ffn, win, drop).to(device)
    p = count_params(m)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=0.01)
    
    m.train()
    s = time.time()
    for step in range(2000):
        x, y = get_batch(train_data, 32, 256)
        opt.zero_grad()
        F.cross_entropy(m(x).view(-1,V), y.view(-1)).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if step % 500 == 0:
            m.eval()
            with torch.no_grad():
                vl = sum(F.cross_entropy(m(get_batch(val_data,32,256)[0]).view(-1,V),
                         get_batch(val_data,32,256)[1].view(-1)).item() for _ in range(10))/10
            m.train()
    
    m.eval()
    with torch.no_grad():
        fl = sum(F.cross_entropy(m(get_batch(val_data,32,256)[0]).view(-1,V),
                 get_batch(val_data,32,256)[1].view(-1)).item() for _ in range(50))/50
    ppl = math.exp(fl)
    
    txt = generate(m, "\nA mesterseges", 150, 0.7)
    print(f"{label:<40} {p:<10} {fl:<10.4f} {ppl:<10.2f} {time.time()-s:<8.1f}")
    print(f"  Generalva: {txt[:100]}")
    print()

print(f"\n=== Eredmeny ===")
print(f"A legjobb modell altal generalt szoveg:")
print(txt[:300])
