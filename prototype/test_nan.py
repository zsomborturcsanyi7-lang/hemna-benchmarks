"""Gyors NaN teszt - 200 lepes"""
import torch, torch.nn as nn, torch.nn.functional as F, sys, time, math, os
import sentencepiece as spm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
sp = spm.SentencePieceProcessor()
sp.Load(r'C:\NeuraNode\bitnet\data\bitnet_pretrain\tokenizer\tokenizer.model')
VOCAB = sp.GetPieceSize()
print(f"Szokincs: {VOCAB}, device: {device}")

# Adat
DATA_DIR = r'C:\NeuraNode\bitnet\data\bitnet_pretrain\tokenized'
d = torch.load(os.path.join(DATA_DIR, 'shard_0.pt'), map_location='cpu', weights_only=True)
train_data = d[:70000]  # egy shard nagy resze
val_data = d[70000:72000]
print(f"Adat: {train_data.shape}")

def get_batch(d, bs=4, bl=512):
    ix = torch.randint(len(d), (bs,))
    x = d[ix, :bl-1].to(device, dtype=torch.long)
    y = d[ix, 1:bl].to(device, dtype=torch.long)
    return x, y

# RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight

# GQA
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
        q = self.wq(x).view(B,T,self.nh,self.hd).transpose(1,2)
        k = self.wk(x).view(B,T,self.nkv,self.hd).transpose(1,2)
        v = self.wv(x).view(B,T,self.nkv,self.hd).transpose(1,2)
        if self.nr > 1:
            k = k[:,:,None].expand(-1,-1,self.nr,-1,-1).reshape(B,self.nh,T,self.hd)
            v = v[:,:,None].expand(-1,-1,self.nr,-1,-1).reshape(B,self.nh,T,self.hd)
        wei = (q @ k.transpose(-2,-1)) * (self.hd**-0.5)
        wei = wei.masked_fill(self.mask[:T,:T]==0, float('-inf'))
        wei = wei - wei.max(-1, keepdim=True)[0]
        wei = F.softmax(wei, dim=-1)
        out = (wei @ v).transpose(1,2).reshape(B,T,-1)
        return self.wo(out)

class GatedFFN(nn.Module):
    def __init__(self, dim, ffn):
        super().__init__()
        self.w1 = nn.Linear(dim, ffn, False)
        self.w2 = nn.Linear(ffn, dim, False)
        self.w3 = nn.Linear(dim, ffn, False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class LM(nn.Module):
    def __init__(self, vocab, dim=768, layers=4, heads=12, ffn=2304):
        super().__init__()
        self.tok = nn.Embedding(vocab, dim)
        self.ln = nn.ModuleList([RMSNorm(dim) for _ in range(layers)])
        self.attn = nn.ModuleList([GQA(dim, heads, max(2, heads//4)) for _ in range(layers)])
        self.ffn = nn.ModuleList([GatedFFN(dim, ffn) for _ in range(layers)])
        self.ln_f = RMSNorm(dim)
        self.out = nn.Linear(dim, vocab, False)
    def forward(self, x):
        x = self.tok(x)
        for i in range(len(self.ln)):
            x = x + self.attn[i](self.ln[i](x))
            x = x + self.ffn[i](x)
        return self.out(self.ln_f(x))

model = LM(VOCAB).to(device)
p = sum(p.numel() for p in model.parameters())
print(f"Parameterek: {p:,}")
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

print("\nGyorsteszt 200 lepes...")
for step in range(200):
    x, y = get_batch(train_data, 4, 512)
    opt.zero_grad()
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, VOCAB), y.view(-1))
    
    # NaN check
    if torch.isnan(loss):
        print(f"  NaN at step {step}!")
        for n, p in model.named_parameters():
            if torch.isnan(p).any():
                print(f"    NaN param: {n}")
            if p.grad is not None and torch.isnan(p.grad).any():
                print(f"    NaN grad: {n}")
        sys.exit(1)
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    
    if step % 50 == 0:
        print(f"  Step {step}: loss={loss.item():.4f}")

print("OK - nincs NaN!")
