"""300M parameter magyar nyelvi modell training"""
import torch, torch.nn as nn, torch.nn.functional as F, sys, time, math, os
import sentencepiece as spm

LOG_FILE = r'C:\NeuraNode\hemna_bench\train_log.txt'
def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    print(msg, flush=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
log(f"Eszkoz: {device}")

# Tokenizer betoltese
sp = spm.SentencePieceProcessor()
sp.Load(r'C:\NeuraNode\bitnet\data\bitnet_pretrain\tokenizer\tokenizer.model')
VOCAB = sp.GetPieceSize()
log(f"Szokincs: {VOCAB}")

# Adat betoltese (1 shard a gyors teszthez, tobb a 300M-hez)
DATA_DIR = r'C:\NeuraNode\bitnet\data\bitnet_pretrain\tokenized'
shards = [torch.load(os.path.join(DATA_DIR, f'shard_{i}.pt'), map_location='cpu', weights_only=True) 
          for i in range(4)]
all_data = torch.cat(shards, dim=0)  # [N, 512]
log(f"Adat: {all_data.shape[0]} szekvencia, {all_data.shape[1]} hossz")
log(f"  Osszes token: {all_data.shape[0] * all_data.shape[1]:,}")

# Train/val split
n = all_data.shape[0]
train_data = all_data[:int(n*0.98)]
val_data = all_data[int(n*0.98):]
log(f"Train: {train_data.shape[0]}, Val: {val_data.shape[0]}")

def get_batch(d, bs=16, bl=512):
    ix = torch.randint(len(d), (bs,))
    x = d[ix, :bl-1].to(device, dtype=torch.long)
    y = d[ix, 1:bl].to(device, dtype=torch.long)
    return x, y

# ============================================================
# 300M modell architektura
# ============================================================
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight

class GQA(nn.Module):
    def __init__(self, dim, nh, nkv):
        super().__init__()
        self.nh, self.nkv, self.hd = nh, nkv, dim // nh
        self.nr = nh // nkv
        self.wq = nn.Linear(dim, nh*self.hd, False)
        self.wk = nn.Linear(dim, nkv*self.hd, False)
        self.wv = nn.Linear(dim, nkv*self.hd, False)
        self.wo = nn.Linear(nh*self.hd, dim, False)
        self.register_buffer('mask', torch.tril(torch.ones(512, 512)))
    
    def forward(self, x):
        B,T,D = x.shape
        x = x.to(torch.float32)  # stabilis szamolas
        q = self.wq(x).view(B,T,self.nh,self.hd).transpose(1,2)
        k = self.wk(x).view(B,T,self.nkv,self.hd).transpose(1,2)
        v = self.wv(x).view(B,T,self.nkv,self.hd).transpose(1,2)
        if self.nr > 1:
            k = k[:,:,None].expand(-1,-1,self.nr,-1,-1).reshape(B,self.nh,T,self.hd)
            v = v[:,:,None].expand(-1,-1,self.nr,-1,-1).reshape(B,self.nh,T,self.hd)
        wei = (q @ k.transpose(-2,-1)) * (self.hd**-0.5)
        # Stabil softmax
        wei = wei.masked_fill(self.mask[:T,:T]==0, float('-inf'))
        wei = wei - wei.max(-1, keepdim=True)[0]  # stabilizacio
        wei = F.softmax(wei, dim=-1)
        out = (wei @ v).transpose(1,2).reshape(B,T,-1).to(x.dtype)
        return self.wo(out)

class GatedFFN(nn.Module):
    def __init__(self, dim, ffn):
        super().__init__()
        self.w1 = nn.Linear(dim, ffn, False)
        self.w2 = nn.Linear(ffn, dim, False)
        self.w3 = nn.Linear(dim, ffn, False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class LM300M(nn.Module):
    def __init__(self, vocab, dim=1024, layers=24, heads=16, ffn=3072):
        super().__init__()
        self.tok = nn.Embedding(vocab, dim)
        self.ln = nn.ModuleList([RMSNorm(dim) for _ in range(layers)])
        self.attn = nn.ModuleList([GQA(dim, heads, max(4, heads//4)) for _ in range(layers)])
        self.ffn = nn.ModuleList([GatedFFN(dim, ffn) for _ in range(layers)])
        self.ln_f = RMSNorm(dim)
        self.out = nn.Linear(dim, vocab, False)  # tied
    
    def forward(self, x):
        x = self.tok(x)
        for i in range(len(self.ln)):
            x = x + self.attn[i](self.ln[i](x))
            x = x + self.ffn[i](x)
        return self.out(self.ln_f(x))
    
    @torch.no_grad()
    def generate(self, prompt_ids, max_len=100, temp=0.7, top_k=50):
        self.eval()
        x = prompt_ids.unsqueeze(0).to(device)
        out = prompt_ids.tolist()
        for _ in range(max_len):
            logits = self(x[:, -512:])[0, -1]
            if top_k > 0:
                vals, _ = torch.topk(logits, top_k)
                logits[logits < vals[-1]] = float('-inf')
            probs = F.softmax(logits / temp, dim=-1)
            nxt = torch.multinomial(probs, 1).item()
            out.append(nxt)
            x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
        return sp.DecodeIds(out)

# ============================================================
# Training
# ============================================================
dim = 768
layers = 6
heads = 12
ffn = 2304

model = LM300M(VOCAB, dim, layers, heads, ffn).to(device)
params = sum(p.numel() for p in model.parameters())
log(f"\nParameterek: {params:,}")

# Batch meret a 8GB VRAM-hoz
bs = 4
total_steps = 50000
log_every = 500
save_every = 5000

opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01, betas=(0.9, 0.95))
warmup_steps = 100

log(f"\nTraining {total_steps} lepes...")
log(f"  Batch: {bs}, hossz: 512, lr: 1e-4")
log(f"  Tokens per step: {bs*511:,}")
log(f"  Total tokens: {bs*511*total_steps:,}")

s = time.time()
for step in range(total_steps):
    model.train()
    x, y = get_batch(train_data, bs, 512)
    
    # Warmup
    if step < warmup_steps:
        lr = 1e-4 * (step + 1) / warmup_steps
        for g in opt.param_groups:
            g['lr'] = lr
    
    opt.zero_grad()
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, VOCAB), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    
    if step % log_every == 0:
        model.eval()
        with torch.no_grad():
            xv, yv = get_batch(val_data, 8, 512)
            vl = F.cross_entropy(model(xv).view(-1, VOCAB), yv.view(-1)).item()
        ppl = math.exp(vl)
        tok_s = 511 * bs * log_every / (time.time() - s + 1e-8)
        log(f"  Step {step:5d}/{total_steps}: train_loss={loss.item():.4f}, val_loss={vl:.4f}, ppl={ppl:.2f}, tok/s={tok_s:.0f}")
        s = time.time()
    
    if step > 0 and step % save_every == 0:
        torch.save(model.state_dict(), 'lm60m_step{step}.pt')
        test = sp.EncodeAsIds("A mesterseges intelligencia")
        txt = model.generate(torch.tensor(test), 100, 0.7)
        log(f"  Generalas: {txt[:150]}")

torch.save(model.state_dict(), 'lm173m_final.pt')
log(f"\nKesz!")
test = sp.EncodeAsIds("A mesterseges intelligencia")
txt = model.generate(torch.tensor(test), 200, 0.7)
log(f"Vegso generalas:\n{txt[:300]}")
log("Training vege.")
