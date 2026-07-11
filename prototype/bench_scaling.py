"""
HEMNA vs Transformer scaling benchmark — karakter szintu nyelvi modell

Osszehasonlitjuk: Standard Transformer FFN vs HEMNA (T0->T1->T2) FFN
Kulonbozo meretekben, es meglatjuk melyik hatekonyabb parameterekben.
"""
import torch, torch.nn as nn, torch.nn.functional as F, sys, time, math, os
sys.path.insert(0, '.')
from hemna_v3 import GrowingLayer, Tier0Linear, Tier1Linear

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# ============================================================
# 1. Adat: kis magyar szoveg
# ============================================================
print("Letoltes...")
# Hasznaljunk egy kis minta szoveget (petofi, arany, stb.)
# Ha nincs, generalunk egy egyszeru szoveget
torch.manual_seed(42)

text = """Az ember agya egy csodalatos szerkezet. Kutatasa mind a mai napig tart.
A nevezo mesterseges intelligencia egy olyan terulet, ahol a szamitogepeket probaljuk 
emberi intelligenciaval felruhazni. A gepi tanulas segitsegevel a szamitogepek kepesek 
mintakat felismerni az adatokbol. A melytanulas tovabb lep: tobb retegu halozatokkal 
dolgozik. A nyelvfeldolgozas az egyik legnehezebb feladat. Egy gepnek meg kell ertenie 
a szavak jelentese kozotti osszefuggeseket. A transformer architektura forradalmasitotta 
a termeszetes nyelvfeldolgozast. Az attention mechanizmus minden szot minden szoval 
osszekot. Igy a gep kepes megtanulni hogy egy mondatban mely szavak fontosak egymas 
szamara. A jovo mesterseges intelligenciaja meg tobbet fog tudni. A kutatok ujabb es 
ujabb modszereket fejlesztenek. A cel egy olyan gep letrehozasa ami tenylegesen 
megerti az emberi nyelvet. Ez az ut hosszu de izgalmas. Minden kis lepes kozelebb 
visz a celhoz. A technologia fejlodese felgyorsitja ezt a folyamatot. A mai nevezesi 
modszerek mar most lenyugozo eredmenyeket mutatnak. A nyelvi modellek keptelenek 
megerteni a kontextust es ertelmes valaszokat adni. A jovo meg tobb meglepetest 
tartogat. A mesterseges intelligencia nem csak a nyelvet de az egesz vilagot 
megvaltoztatja. Az emberiseg elott all a lehetoseg hogy egy intelligens partnert 
hozzon letre. A feladat nem konnyu de a cel  ertekes. Mindenki hozzatehet valamit 
a fejlodeshez. A tudomany nyitott mindenki szamara.""" * 50  # ~100K karakter

chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {c:i for i,c in enumerate(chars)}
idx_to_char = {i:c for i,c in enumerate(chars)}

print(f"Szoveg hossza: {len(text)} karakter")
print(f"Szokincs: {vocab_size}")

def encode(s):
    return torch.tensor([char_to_idx[c] for c in s], dtype=torch.long)

def decode(t):
    return ''.join(idx_to_char[i.item()] for i in t)

data = encode(text)
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

def get_batch(data, batch_size=64, block_size=64):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# ============================================================
# 2. Standard Transformer FFN
# ============================================================
class TransformerFFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, dim))
    def forward(self, x):
        return self.net(x)

# ============================================================
# 3. HEMNA FFN (T0->T1->T2)
# ============================================================
class HEMNAFFN(nn.Module):
    """A standard FFN helyett egy HEMNA reteg."""
    def __init__(self, dim, max_neurons):
        super().__init__()
        self.dim = dim
        # Bemeneti linear: dim -> growing layer
        self.in_proj = nn.Linear(dim, max_neurons)
        # Growing layer: T0->T1->T2
        self.grow = GrowingLayer(max_neurons, max_neurons, 
                                  grad_threshold_t1=0.001, grad_threshold_t2=0.001,
                                  patience=20)
        # Kimeneti linear: growing layer -> dim
        self.out_proj = nn.Linear(max_neurons, dim)
        # Dropout a szabalyozasert
        self.drop = nn.Dropout(0.1)
    
    def forward(self, x):
        # x: [B, T, dim]
        B, T, D = x.shape
        h = self.in_proj(x)  # [B, T, max_neurons]
        
        # Growing layer minden tokenre kulon
        h_flat = h.reshape(B*T, -1)
        h_out = self.grow(h_flat)
        
        out = self.out_proj(h_out).reshape(B, T, D)
        return self.drop(out)

# ============================================================
# 4. Mini Transformer
# ============================================================
class Head(nn.Module):
    """Egy attention fej."""
    def __init__(self, dim, head_size):
        super().__init__()
        self.key = nn.Linear(dim, head_size, bias=False)
        self.query = nn.Linear(dim, head_size, bias=False)
        self.value = nn.Linear(dim, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(64, 64)))
        self.drop = nn.Dropout(0.1)
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.drop(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(dim, head_size) for _ in range(n_head)])
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(0.1)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.drop(self.proj(out))

class Block(nn.Module):
    def __init__(self, dim, n_head, ffn_fn):
        super().__init__()
        head_size = dim // n_head
        self.sa = MultiHeadAttention(dim, n_head, head_size)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = ffn_fn
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, dim, n_layers, n_head, ffn_fn):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.pos_embedding = nn.Embedding(64, dim)
        self.blocks = nn.Sequential(*[Block(dim, n_head, ffn_fn) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size)
    
    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.pos_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.lm_head(x)

# ============================================================
# 5. Benchmark
# ============================================================
def count_params(m):
    return sum(p.numel() for p in m.parameters())

def estimate_loss(model, eval_iters=20):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(eval_iters):
            x, y = get_batch(val_data, 64, 64)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            losses.append(loss.item())
    model.train()
    return sum(losses)/len(losses)

# Konfiguraciok
configs = [
    # (dim, n_head, n_layers, ffn_dim/neurons, tipus)
    # Standard Transformer
    (32, 2, 2, 64, 'TF'),
    (64, 4, 2, 128, 'TF'),
    (64, 4, 3, 256, 'TF'),
    (128, 4, 2, 256, 'TF'),
    # HEMNA
    (32, 2, 2, 64, 'HEMNA'),
    (64, 4, 2, 128, 'HEMNA'),
    (64, 4, 3, 256, 'HEMNA'),
    (128, 4, 2, 256, 'HEMNA'),
]

results = []

print()
print(f"{'Tipus':<8} {'Dim':<5} {'FFN':<7} {'Params':<10} {'Loss':<10} {'Perplexity':<12}")
print("-" * 55)

for dim, n_head, n_layers, ffn_val, tipus in configs:
    torch.manual_seed(42)
    
    if tipus == 'TF':
        ffn_fn = TransformerFFN(dim, ffn_val)
    else:
        # HEMNA: sajat modul, nem closure
        class HEMBlockFFN(nn.Module):
            def __init__(self_):
                super().__init__()
                self_.in_proj = nn.Linear(dim, ffn_val)
                self_.grow = GrowingLayer(ffn_val, ffn_val, 0.001, 0.001, 20)
                self_.out_proj = nn.Linear(ffn_val, dim)
                self_.drop = nn.Dropout(0.1)
            def forward(self_, x):
                B,T,D = x.shape
                h = self_.in_proj(x)
                h = self_.grow(h.reshape(B*T, -1)).reshape(B, T, ffn_val)
                return self_.drop(self_.out_proj(h))
        ffn_fn = HEMBlockFFN()
    
    elif tipus == 'MAMBA':
        # Mamba blokk: allapotter modell a soros feldolgozashoz
        class MambaBlock(nn.Module):
            def __init__(self_):
                super().__init__()
                state_dim = max(4, dim // 8)  # allapot meret
                self_.state_dim = state_dim
                # Bemenet projektcio
                self_.in_proj = nn.Linear(dim, dim * 2)  # x es z
                # Selective parameters: B es C input-fuggo
                self_.b_proj = nn.Linear(dim, state_dim)
                self_.c_proj = nn.Linear(dim, state_dim)
                # Delta (lepeskoz): tanult, input-fuggo
                self_.delta_proj = nn.Linear(dim, 1)
                # A: tanult allapotmatrix (diagonalis)
                self_.log_A = nn.Parameter(torch.randn(state_dim) * 0.1)
                # Kimenet projektcio
                self_.out_proj = nn.Linear(dim, dim)
                self_.drop = nn.Dropout(0.1)
            
            def forward(self_, x):
                B, T, D = x.shape
                sd = self_.state_dim
                
                # x es z: [B, T, D]
                xz = self_.in_proj(x)
                x_lin, z = xz.chunk(2, dim=-1)  # x, z: [B, T, D]
                
                # delta: [B, T, 1]
                delta = F.softplus(self_.delta_proj(x)) + 0.001
                
                # B: [B, T, sd], C: [B, T, sd]
                B = self_.b_proj(x_lin)
                C = self_.c_proj(x_lin)
                
                # A: diagonalis, negativ (stabilis)
                A = -torch.exp(self_.log_A)  # [sd]
                
                # Diszkretizalos: A_bar = exp(delta * A), B_bar = delta * B
                # A_bar: [B, T, sd] minden allapothoz
                A_bar = torch.exp(delta * A.unsqueeze(0).unsqueeze(0))  # [B, T, sd]
                B_bar = delta * B  # [B, T, sd]
                
                # Szekvencialis scan: h_t = A_bar_t * h_{t-1} + B_bar_t * x_lin_t
                h = torch.zeros(B, sd, device=x.device)
                y = torch.zeros_like(x_lin)
                for t in range(T):
                    # h: [B, sd]
                    h = A_bar[:, t, :] * h + B_bar[:, t, :] * x_lin[:, t, :]
                    # y: [B, D] = C * h
                    y[:, t, :] = C[:, t, :] @ h.unsqueeze(-1) ??
                
                # Hat ez igy nem jo...
                pass
            
        ffn_fn = None  # TODO
    
    model = MiniTransformer(vocab_size, dim, n_layers, n_head, ffn_fn).to(device)
    n_params = count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Training
    model.train()
    for step in range(500):
        x, y = get_batch(train_data, 64, 64)
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        # HEMNA: minden lepesben growth
        if tipus == 'HEMNA':
            ffn_fn.grow.update_growth()
        
        if step % 100 == 0:
            val_loss = estimate_loss(model, 10)
    
    final_loss = estimate_loss(model, 50)
    ppl = math.exp(final_loss)
    print(f"{tipus:<8} {dim:<5} {str(ffn_val):<7} {n_params:<10} {final_loss:<10.4f} {ppl:<12.2f}")
    results.append((tipus, n_params, final_loss, ppl))

print(f"\n=== OSSZEGZES ===")
# Legjobb TF es HEMNA
best_tf = min([r for r in results if r[0]=='TF'], key=lambda x: x[2])
best_he = min([r for r in results if r[0]=='HEMNA'], key=lambda x: x[2])
print(f"Legjobb Transformer: {best_tf[2]:.4f} loss ({best_tf[1]} param)")
print(f"Legjobb HEMNA:      {best_he[2]:.4f} loss ({best_he[1]} param)")

# Parameterek szerint osszehasonlitas
print(f"\nParameter hatekonysag:")
for tf, he in zip([r for r in results if r[0]=='TF'], [r for r in results if r[0]=='HEMNA']):
    diff = tf[2] - he[2]
    winner = "HEMNA" if diff > 0 else "TF"
    print(f"  {tf[1]:<10} param: TF={tf[2]:.4f} vs HEMNA={he[2]:.4f} -> {winner} nyer {abs(diff):.4f}")
