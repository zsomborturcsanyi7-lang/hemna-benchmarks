"""120M modell teszt - szoveg generalas"""
import torch, torch.nn as nn, torch.nn.functional as F, sys, time, os
import sentencepiece as spm

LOG_FILE = r'C:\NeuraNode\hemna_bench\test_120m_out.txt'
def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Eszkoz: {device}")

# Tokenizer
sp = spm.SentencePieceProcessor()
sp.Load(r'C:\NeuraNode\bitnet\data\bitnet_pretrain\tokenizer\tokenizer.model')
V = sp.GetPieceSize()
print(f"Szokincs: {V}")

# Model architecture
class RMSNorm(nn.Module):
    def __init__(self, dim): super().__init__(); self.w = nn.Parameter(torch.ones(dim))
    def forward(self, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.w

class GQA(nn.Module):
    def __init__(self, dim, nh, nkv):
        super().__init__()
        self.nh, self.nkv, self.hd, self.nr = nh, nkv, dim//nh, nh//nkv
        self.wq = nn.Linear(dim, nh*self.hd, False)
        self.wk = nn.Linear(dim, nkv*self.hd, False)
        self.wv = nn.Linear(dim, nkv*self.hd, False)
        self.wo = nn.Linear(nh*self.hd, dim, False)
        self.register_buffer('m', torch.tril(torch.ones(512, 512)))
    def forward(self, x):
        B,T,D = x.shape
        q = self.wq(x).view(B,T,self.nh,self.hd).transpose(1,2)
        k = self.wk(x).view(B,T,self.nkv,self.hd).transpose(1,2)
        v = self.wv(x).view(B,T,self.nkv,self.hd).transpose(1,2)
        if self.nr > 1:
            k = k[:,:,None].expand(-1,-1,self.nr,-1,-1).reshape(B,self.nh,T,self.hd)
            v = v[:,:,None].expand(-1,-1,self.nr,-1,-1).reshape(B,self.nh,T,self.hd)
        w = (q @ k.transpose(-2,-1)) * (self.hd**-0.5)
        w = w.masked_fill(self.m[:T,:T]==0, float('-inf'))
        w = w - w.max(-1, keepdim=True)[0]
        return self.wo((F.softmax(w, dim=-1) @ v).transpose(1,2).reshape(B,T,-1))

class FFN(nn.Module):
    def __init__(self, dim, h):
        super().__init__()
        self.w1 = nn.Linear(dim, h, False); self.w2 = nn.Linear(h, dim, False); self.w3 = nn.Linear(dim, h, False)
    def forward(self, x): return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Block(nn.Module):
    def __init__(self, dim, nh, nkv, ffn):
        super().__init__()
        self.ln1 = RMSNorm(dim); self.ln2 = RMSNorm(dim)
        self.attn = GQA(dim, nh, nkv); self.ffn = FFN(dim, ffn)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class LM(nn.Module):
    def __init__(self, V, dim, layers, heads, ffn):
        super().__init__()
        self.tok = nn.Embedding(V, dim)
        self.blocks = nn.ModuleList([Block(dim, heads, max(2, heads//4), ffn) for _ in range(layers)])
        self.ln_f = RMSNorm(dim)
        self.out = nn.Linear(dim, V, False)
    def forward(self, x):
        x = self.tok(x)
        for b in self.blocks: x = b(x)
        return self.out(self.ln_f(x))
    @torch.no_grad()
    def generate(self, prompt, n=200, t=0.7, k=50):
        self.eval()
        x = prompt.unsqueeze(0).to(device)
        out = prompt.tolist()
        for _ in range(n):
            l = self(x[:,-512:])[0,-1]
            if k > 0: v,_ = torch.topk(l, k); l[l < v[-1]] = float('-inf')
            out.append(torch.multinomial(F.softmax(l/t, dim=-1), 1).item())
            x = torch.cat([x, torch.tensor([[out[-1]]], device=device)], dim=1)
        return sp.DecodeIds(out)

# Load 120M modell
print("\n120M modell betoltese...")
model = LM(V, 768, 12, 12, 2304).to(device)
state = torch.load(r'C:\Users\neura\lm120m_clean_final.pt', map_location=device, weights_only=True)
model.load_state_dict(state)
model.eval()
p = sum(p.numel() for p in model.parameters())
print(f"Parameterek: {p:,}")
print(f"Model meret: {sum(p.numel()*4 for p in model.parameters())/1024**3:.2f} GB (FP32)")

# Teszt promptok
tests = [
    "A mesterseges intelligencia",
    "Magyarorszag fovarosa",
    "A programozas nyelv",
    "A macska es a kutya",
]

log("="*60)
log("120M MODELL TESZT")
log("="*60)

for prompt in tests:
    log(f"\n--- Prompt: {prompt} ---")
    ids = torch.tensor(sp.EncodeAsIds(prompt))
    s = time.time()
    text = model.generate(ids, n=150, t=0.7, k=50)
    elapsed = time.time() - s
    log(f"  [{elapsed:.1f}s] {text[:300]}")
    log("")
