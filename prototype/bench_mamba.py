"""Transformer vs Mamba scaling benchmark"""
import torch, torch.nn as nn, torch.nn.functional as F, sys, time, math
sys.path.insert(0, '.')
from hemna_v3 import GrowingLayer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# ============================================================
# 1. Adat
# ============================================================
torch.manual_seed(42)
text = """Az ember agya egy csodalatos szerkezet. Kutatasa mind a mai napig tart.
A mesterseges intelligencia egy olyan terulet ahol a szamitogepeket probaljuk 
emberi intelligenciaval felruhazni. A gepi tanulas segitsegevel a szamitogepek 
kepesek mintakat felismerni az adatokbol. A melytanulas tovabb lep tobb retegu 
halozatokkal dolgozik. A nyelvfeldolgozas az egyik legnehezebb feladat. 
Egy gepnek meg kell ertenie a szavak jelentese kozotti osszefuggeseket. 
A transformer architektura forradalmasitotta a termeszetes nyelvfeldolgozast. 
Az attention mechanizmus minden szot minden szoval osszekot. 
A jovo mesterseges inteligenciaja meg tobbet fog tudni.""" * 100

chars = sorted(list(set(text)))
vocab_size = len(chars)
c2i = {c:i for i,c in enumerate(chars)}
i2c = {i:c for i,c in enumerate(chars)}
data = torch.tensor([c2i[c] for c in text], dtype=torch.long)
n = len(data)
train_data, val_data = data[:int(n*0.9)], data[int(n*0.9):]
print(f"Szoveg: {len(text)} kar, szokincs: {vocab_size}")

def get_batch(data, bs=64, bl=64):
    ix = torch.randint(len(data)-bl, (bs,))
    x = torch.stack([data[i:i+bl] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+bl+1] for i in ix]).to(device)
    return x, y

def count_params(m):
    return sum(p.numel() for p in m.parameters())

# ============================================================
# 2. Transformer
# ============================================================
class Head(nn.Module):
    def __init__(self, dim, hs):
        super().__init__()
        self.k = nn.Linear(dim, hs, False)
        self.q = nn.Linear(dim, hs, False)
        self.v = nn.Linear(dim, hs, False)
        self.register_buffer('t', torch.tril(torch.ones(64, 64)))
    def forward(self, x):
        B,T,C = x.shape
        wei = self.q(x) @ self.k(x).transpose(-2,-1) * (C**-0.5)
        wei = wei.masked_fill(self.t[:T,:T]==0, float('-inf'))
        return F.softmax(wei, dim=-1) @ self.v(x)

class MHA(nn.Module):
    def __init__(self, dim, nh):
        super().__init__()
        hs = dim // nh
        self.heads = nn.ModuleList([Head(dim, hs) for _ in range(nh)])
        self.proj = nn.Linear(dim, dim)
    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))

class TFBlock(nn.Module):
    def __init__(self, dim, nh, ffn_dim):
        super().__init__()
        self.sa = MHA(dim, nh)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.ReLU(), nn.Linear(ffn_dim, dim))
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class TF(nn.Module):
    def __init__(self, dim, nh, n_l, ffn_dim):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(64, dim)
        self.blocks = nn.Sequential(*[TFBlock(dim, nh, ffn_dim) for _ in range(n_l)])
        self.ln = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, vocab_size)
    def forward(self, x):
        B,T = x.shape
        x = self.tok(x) + self.pos(torch.arange(T, device=device))
        x = self.blocks(x)
        return self.out(self.ln(x))

# ============================================================
# 3. Mamba (egyszerusitett)
# ============================================================
class MambaBlock(nn.Module):
    """Allapotter modell soros feldolgozassal (Mamba-szeru)."""
    def __init__(self, dim, state_dim=8):
        super().__init__()
        self.sd = state_dim
        self.in_proj = nn.Linear(dim, dim)
        self.x_proj = nn.Linear(dim, state_dim)  # D -> sd a scan-hez
        self.b_proj = nn.Linear(dim, state_dim)
        self.c_proj = nn.Linear(dim, state_dim)
        self.delta_proj = nn.Linear(dim, 1)
        self.log_A = nn.Parameter(torch.randn(state_dim) * 0.1)
        self.out_proj = nn.Linear(state_dim, dim)  # sd -> D vissza
        self.ln = nn.LayerNorm(dim)
    
    def forward(self, x):
        """x: [B, T, D] -> [B, T, D]"""
        B, T, D = x.shape
        sd = self.sd
        
        x_norm = self.ln(x)
        x_lin = self.in_proj(x_norm)  # [B, T, D]
        
        # Input a scan-hez: D -> sd
        x_for_scan = self.x_proj(x_lin)  # [B, T, sd]
        
        # Selective parameters
        delta = F.softplus(self.delta_proj(x_lin)) + 0.001  # [B, T, 1]
        B_mat = self.b_proj(x_lin)  # [B, T, sd]
        C_mat = self.c_proj(x_lin)  # [B, T, sd]
        
        # Stabil A
        A = -torch.exp(self.log_A)  # [sd]
        
        # Diszkretizacio
        A_bar = torch.exp(delta * A.view(1, 1, sd))
        B_bar = delta * B_mat
        
        # Scan: h_t = A_bar * h_{t-1} + B_bar * x_for_scan_t
        h = torch.zeros(B, sd, device=x.device)
        y_sd = torch.zeros(B, T, sd, device=x.device)
        for t in range(T):
            h = A_bar[:, t, :] * h + B_bar[:, t, :] * x_for_scan[:, t, :]
            y_sd[:, t, :] = C_mat[:, t, :] * h  # [B, sd]
        
        # Vissza sd -> D
        y = self.out_proj(y_sd)  # [B, T, D]
        return y  # nincs residual, a fuggohivo adja hozza

class Mamba(nn.Module):
    """Tiszta Mamba (nincs attention)."""
    def __init__(self, dim, n_l, state_dim=8):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(64, dim)
        self.blocks = nn.ModuleList([MambaBlock(dim, state_dim) for _ in range(n_l)])
        self.ln = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, vocab_size)
    
    def forward(self, x):
        B,T = x.shape
        x = self.tok(x) + self.pos(torch.arange(T, device=device))
        for block in self.blocks:
            x = x + block(x)  # residual
        return self.out(self.ln(x))

# ============================================================
# 4. Benchmark
# ============================================================
configs = [
    # Transformer: (dim, n_head, n_layers, ffn_dim)
    ('TF', 32, 2, 2, 64),
    ('TF', 64, 4, 2, 128),
    ('TF', 64, 4, 3, 256),
    ('TF', 128, 4, 2, 256),
    # Mamba: (dim, n_layers, state_dim)
    ('MAMBA', 32, 2, 2, None),  # dim, dummy, layers, state_dim
    ('MAMBA', 64, 4, 2, None),
    ('MAMBA', 64, 4, 3, None),
    ('MAMBA', 128, 4, 2, None),
]

print(f"\n{'Tipus':<8} {'Dim':<5} {'Extra':<7} {'Params':<10} {'Loss':<10} {'PPL':<12}")
print("-" * 55)

results = []
for tipus, dim, nh, nl, extra in configs:
    torch.manual_seed(42)
    
    if tipus == 'TF':
        model = TF(dim, nh, nl, extra).to(device)
    else:
        model = Mamba(dim, nl, dim // 8 + 4).to(device)
    
    p = count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    model.train()
    for step in range(500):
        x, y = get_batch(train_data, 64, 64)
        opt.zero_grad()
        loss = F.cross_entropy(model(x).view(-1, vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            model.eval()
            with torch.no_grad():
                vl = sum(F.cross_entropy(model(get_batch(val_data, 64, 64)[0]).view(-1, vocab_size), 
                         get_batch(val_data, 64, 64)[1].view(-1)).item() for _ in range(10)) / 10
            model.train()
    
    model.eval()
    with torch.no_grad():
        losses = []
        for _ in range(50):
            x, y = get_batch(val_data, 64, 64)
            losses.append(F.cross_entropy(model(x).view(-1, vocab_size), y.view(-1)).item())
    fl = sum(losses)/len(losses)
    ppl = math.exp(fl)
    print(f"{tipus:<8} {dim:<5} {str(extra or dim//8+4):<7} {p:<10} {fl:<10.4f} {ppl:<12.2f}")
    results.append((tipus, p, fl, ppl))

print(f"\n=== OSSZEGZES ===")
for tipus in ['TF', 'MAMBA']:
    best = min([r for r in results if r[0]==tipus], key=lambda x: x[2])
    print(f"Legjobb {tipus}: {best[3]:.2f} perplexity ({best[1]} param)")

print(f"\nParameter hatekonysag:")
tf_results = [r for r in results if r[0]=='TF']
m_results = [r for r in results if r[0]=='MAMBA']
for i, (tf, m) in enumerate(zip(tf_results, m_results)):
    diff = m[2] - tf[2]
    w = "TF" if diff > 0 else "MAMBA"
    print(f"  ~{tf[1]:<10} param: TF={tf[3]:.2f} vs MAMBA={m[3]:.2f} -> {w} nyer {abs(diff):.2f}")
