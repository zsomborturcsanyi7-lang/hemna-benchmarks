"""Magyar nyelvi LM training - javitott valtozat
- Csak fajlba logolas (nincs print blokkolas)
- Gradient checkpointing
- Valaszthato meret"""
import torch, torch.nn as nn, torch.nn.functional as F, time, math, os, sys
import sentencepiece as spm

LOG = r'C:\NeuraNode\hemna_bench\train_v2_log.txt'
def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
log(f"Device: {device}")

# Tokenizer
sp = spm.SentencePieceProcessor()
sp.Load(r'C:\NeuraNode\bitnet\data\bitnet_pretrain\tokenizer\tokenizer.model')
V = sp.GetPieceSize()
log(f"Vocab: {V}")

# Adat
import glob
DATA_DIR = r'C:\NeuraNode\hemna_bench\combined_data'
shards = [torch.load(os.path.join(DATA_DIR, f'combined_shard_{i}.pt'), map_location='cpu', weights_only=True) for i in range(105)]
data = torch.cat(shards, dim=0)
log(f"Data: {data.shape[0]} seq, {data.shape[0]*data.shape[1]:,} tokens")
train_d, val_d = data[:int(len(data)*0.98)], data[int(len(data)*0.98):]

def get_batch(d, bs, bl):
    ix = torch.randint(len(d), (bs,))
    x = d[ix, :bl-1].to(device, dtype=torch.long)
    y = d[ix, 1:bl].to(device, dtype=torch.long)
    return x, y

# Architektura
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
    def __init__(self, dim, nh, nkv, ffn, use_ckpt=False):
        super().__init__()
        self.ln1 = RMSNorm(dim); self.ln2 = RMSNorm(dim)
        self.attn = GQA(dim, nh, nkv); self.ffn = FFN(dim, ffn)
        self.use_ckpt = use_ckpt
    def forward(self, x):
        if self.use_ckpt and self.training:
            x = x + torch.utils.checkpoint.checkpoint(self._attn_pass, self.ln1(x))
            x = x + torch.utils.checkpoint.checkpoint(self._ffn_pass, self.ln2(x))
        else:
            x = x + self.attn(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
        return x
    def _attn_pass(self, x): return self.attn(x)
    def _ffn_pass(self, x): return self.ffn(x)

class LM(nn.Module):
    def __init__(self, V, dim, layers, heads, ffn, use_ckpt=False):
        super().__init__()
        self.tok = nn.Embedding(V, dim)
        self.blocks = nn.ModuleList([Block(dim, heads, max(2, heads//4), ffn, use_ckpt) for _ in range(layers)])
        self.ln_f = RMSNorm(dim)
        self.out = nn.Linear(dim, V, False)
    def forward(self, x):
        x = self.tok(x)
        for b in self.blocks:
            x = b(x)
        return self.out(self.ln_f(x))
    @torch.no_grad()
    def generate(self, prompt, n=100, t=0.7, k=50):
        self.eval()
        x = prompt.unsqueeze(0).to(device)
        out = prompt.tolist()
        for _ in range(n):
            l = self(x[:,-512:])[0,-1]
            if k > 0:
                v,_ = torch.topk(l, k); l[l < v[-1]] = float('-inf')
            out.append(torch.multinomial(F.softmax(l/t, dim=-1), 1).item())
            x = torch.cat([x, torch.tensor([[out[-1]]], device=device)], dim=1)
        return sp.DecodeIds(out)

# ============================================================
# Konfiguracio - valtoztathato
# ============================================================
# Preset-ek:
#   60M:    dim=768,  layers=6,  heads=12, ffn=2304  (~60M)
#   120M:   dim=768,  layers=12, heads=12, ffn=2304  (~120M)
#   180M:   dim=1024, layers=12, heads=16, ffn=3072  (~180M)
#   300M:   dim=1024, layers=24, heads=16, ffn=3072  (~354M) - TDR veszely!

# Valassz egy preset-et:
PRESET = '300M'  # Kombinalt adaton (2.64B token) 300M modell
DATA_DIR = r'C:\NeuraNode\hemna_bench\combined_data'

PRESETS = {
    '60M':  {'dim':768,  'layers':6,  'heads':12, 'ffn':2304, 'bs':4,  'bl':512, 'ckpt':False, 'name':'lm60m_v2'},
    '120M': {'dim':768,  'layers':12, 'heads':12, 'ffn':2304, 'bs':4,  'bl':512, 'ckpt':True,  'name':'lm120m_clean'},
    '180M': {'dim':1024, 'layers':12, 'heads':16, 'ffn':3072, 'bs':4,  'bl':256, 'ckpt':True,  'name':'lm180m'},
    '300M': {'dim':1024, 'layers':24, 'heads':16, 'ffn':3072, 'bs':2,  'bl':256, 'ckpt':True,  'name':'lm300m'},
}

cfg = PRESETS[PRESET]
model = LM(V, cfg['dim'], cfg['layers'], cfg['heads'], cfg['ffn'], cfg['ckpt']).to(device)
p = sum(p.numel() for p in model.parameters())
log(f"Preset: {PRESET} | Params: {p:,} | dim={cfg['dim']}, layers={cfg['layers']}")
log(f"Batch: {cfg['bs']}, seq: {cfg['bl']}, ckpt: {cfg['ckpt']}")

opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01, betas=(0.9, 0.95))

STEPS = 50000
LOG_EVERY = 500
SAVE_EVERY = 5000
log(f"Steps: {STEPS}, log every: {LOG_EVERY}")

s = time.time()
for step in range(STEPS):
    model.train()
    x, y = get_batch(train_d, cfg['bs'], cfg['bl'])
    opt.zero_grad()
    loss = F.cross_entropy(model(x).view(-1, V), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    
    if step % LOG_EVERY == 0:
        model.eval()
        with torch.no_grad():
            xv, yv = get_batch(val_d, 4, 512)
            vl = F.cross_entropy(model(xv).view(-1, V), yv.view(-1)).item()
        tok_s = cfg['bs'] * (cfg['bl']-1) * LOG_EVERY / (time.time() - s + 1e-8)
        log(f"Step {step:5d}/{STEPS}: train={loss.item():.4f} val={vl:.4f} ppl={math.exp(vl):.1f} tok/s={tok_s:.0f}")
        s = time.time()
    
    if step > 0 and step % SAVE_EVERY == 0:
        torch.save(model.state_dict(), f"{cfg['name']}_step{step}.pt")
        txt = model.generate(torch.tensor(sp.EncodeAsIds("A mesterseges intelligencia")), 100, 0.7)
        log(f"  Generated: {txt[:150]}")

torch.save(model.state_dict(), f"{cfg['name']}_final.pt")
log(f"Saving {cfg['name']}_final.pt")
txt = model.generate(torch.tensor(sp.EncodeAsIds("A mesterseges intelligencia")), 200, 0.7)
log(f"FINAL: {txt[:300]}")
log("DONE")
