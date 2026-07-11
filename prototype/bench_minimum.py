"""Minimum parameter count for language acquisition - scaling test"""
import torch, torch.nn as nn, torch.nn.functional as F, sys, time, math
sys.path.insert(0, '.')
from hemna_v3 import Tier0Linear, Tier1Linear

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# ============================================================
# Adat
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
A jovo mesterseges inteligenciaja meg tobbet fog tudni.
A kutatok ujabb es ujabb modszereket fejlesztenek. 
A cel egy olyan gep letrehozasa ami tenylegesen megerti az emberi nyelvet. 
Ez az ut hosszu de izgalmas. Minden kis lepes kozelebb visz a celhoz.""" * 100

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
# Optimizalt mikro architektura
# ============================================================
class GQA(nn.Module):
    """Grouped Query Attention."""
    def __init__(self, dim, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads  # hany query per kv
        self.head_dim = dim // n_heads
        
        self.wq = nn.Linear(dim, n_heads * self.head_dim, False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, False)
        self.register_buffer('mask', torch.tril(torch.ones(256, 256)))
    
    def forward(self, x):
        B, T, D = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        # Repeat KV heads
        if self.n_rep > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).reshape(B, -1, T, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).reshape(B, -1, T, self.head_dim)
        
        wei = q @ k.transpose(-2, -1) * (self.head_dim ** -0.5)
        wei = wei.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        out = (wei @ v).transpose(1, 2).reshape(B, T, -1)
        return self.wo(out)

class GatedFFN(nn.Module):
    """Gated FFN (SwiGLU)."""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, False)
        self.w2 = nn.Linear(hidden_dim, dim, False)
        self.w3 = nn.Linear(dim, hidden_dim, False)
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MicroBlock(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, ffn_dim):
        super().__init__()
        self.sa = GQA(dim, n_heads, n_kv_heads)
        self.ffn = GatedFFN(dim, ffn_dim)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class MicroLM(nn.Module):
    """Optimalizalt mikro nyelvi modell."""
    def __init__(self, vocab_size, dim, n_layers, n_heads, ffn_dim, tied=True):
        super().__init__()
        self.dim = dim
        # Tied embedding
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.Sequential(*[
            MicroBlock(dim, n_heads, max(1, n_heads//4), ffn_dim) for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(dim)
        # Kimenet ugyanaz az embedding matrix (tied)
        self.out_proj = nn.Linear(dim, vocab_size, False)
        # ALiBi: nincs pozicio embedding!
    
    def forward(self, x):
        B, T = x.shape
        x = self.tok_emb(x)
        x = self.blocks(x)
        x = self.ln(x)
        return self.out_proj(x)

# ============================================================
# Scaling benchmark
# ============================================================
# Különböző méretek a paraméterekben
configs = [
    # (dim, layers, heads, ffn_dim, label)
    (16, 2, 2, 32, "16d-2l"),      # ~10K
    (32, 2, 2, 64, "32d-2l"),      # ~30K
    (32, 4, 4, 64, "32d-4l"),      # ~60K
    (64, 2, 4, 128, "64d-2l"),     # ~90K
    (64, 4, 4, 128, "64d-4l"),     # ~170K
    (64, 6, 4, 128, "64d-6l"),     # ~250K
    (128, 2, 4, 256, "128d-2l"),   # ~290K
    (128, 4, 4, 256, "128d-4l"),   # ~550K
]

def generate(model, prompt="\n", max_len=200, temp=0.8):
    model.eval()
    with torch.no_grad():
        x = torch.tensor([c2i[c] for c in prompt], device=device).unsqueeze(0)
        out = []
        for _ in range(max_len):
            logits = model(x[:, -64:])
            probs = F.softmax(logits[0, -1] / temp, dim=-1)
            next_char = torch.multinomial(probs, 1)
            out.append(i2c[next_char.item()])
            x = torch.cat([x, next_char.unsqueeze(0)], dim=1)
        return ''.join(out)

print(f"\n{'Konfig':<12} {'Params':<10} {'Loss':<10} {'PPL':<10} {'Generalt szoveg (elso 50 kar)':<50}")
print("-" * 100)

results = []
for dim, n_l, n_h, ffn_dim, label in configs:
    torch.manual_seed(42)
    n_kv_h = max(1, n_h // 4)
    
    model = MicroLM(vocab_size, dim, n_l, n_h, ffn_dim, tied=True).to(device)
    n_p = count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    
    # Training
    model.train()
    for step in range(1000):
        x, y = get_batch(train_data, 64, 64)
        opt.zero_grad()
        loss = F.cross_entropy(model(x).view(-1, vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 200 == 0:
            model.eval()
            with torch.no_grad():
                vl = sum(F.cross_entropy(model(get_batch(val_data, 64, 64)[0]).view(-1, vocab_size),
                         get_batch(val_data, 64, 64)[1].view(-1)).item() for _ in range(10)) / 10
            model.train()
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        losses = []
        for _ in range(50):
            x, y = get_batch(val_data, 64, 64)
            losses.append(F.cross_entropy(model(x).view(-1, vocab_size), y.view(-1)).item())
    fl = sum(losses)/len(losses)
    ppl = math.exp(fl)
    
    # Generate sample text
    sample = generate(model, "\nA mesterseges", 100, 0.8)
    
    print(f"{label:<12} {n_p:<10} {fl:<10.4f} {ppl:<10.2f} {sample[:60]}")
    results.append((label, n_p, fl, ppl, sample))

print(f"\n=== OSSZEGZES ===")
print(f"A legkisebb modell ami ertelmes szoveget general:")
for label, n_p, fl, ppl, sample in results:
    # Szamold meg hany magyar szo van a mintaban (kb.)
    words = sample.split()
    magyar_szavak = sum(1 for w in words if any(c in 'aáeéiíoóöőuúüű' for c in w.lower()))
    ertelmes = "Lehet" if magyar_szavak >= 3 else "Nem"
    print(f"  {label:<12} {n_p:<10} param: PPL={ppl:.2f}, szo={magyar_szavak}, ertelmes? {ertelmes}")
