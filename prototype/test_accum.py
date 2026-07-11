"""Quick test: does gradient accumulation work with the 300M model?"""
import torch, time, sys

V = 32000  # tokenizer vocab size
device = 'cuda'

# Minimal architektura a teszthez
class RMSNorm(torch.nn.Module):
    def __init__(self, dim): super().__init__(); self.w = torch.nn.Parameter(torch.ones(dim))
    def forward(self, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.w

class GQA(torch.nn.Module):
    def __init__(self, dim, nh, nkv):
        super().__init__()
        self.nh, self.nkv, self.hd, self.nr = nh, nkv, dim//nh, nh//nkv
        self.wq = torch.nn.Linear(dim, nh*self.hd, False)
        self.wk = torch.nn.Linear(dim, nkv*self.hd, False)
        self.wv = torch.nn.Linear(dim, nkv*self.hd, False)
        self.wo = torch.nn.Linear(nh*self.hd, dim, False)
        self.register_buffer('m', torch.tril(torch.ones(512, 512)))
    def forward(self, x):
        B,T,D = x.shape
        q = self.wq(x).view(B,T,self.nh,self.hd).transpose(1,2)
        k = self.wk(x).view(B,T,self.nkv,self.hd).transpose(1,2)
        v = self.wv(x).view(B,T,self.nkv,self.hd).transpose(1,2)
        if self.nh > self.nkv:
            k = k[:,:,None].expand(-1,-1,self.nh//self.nkv,-1,-1).reshape(B,self.nh,T,self.hd)
            v = v[:,:,None].expand(-1,-1,self.nh//self.nkv,-1,-1).reshape(B,self.nh,T,self.hd)
        w = (q @ k.transpose(-2,-1)) * (self.hd**-0.5)
        w = w.masked_fill(self.m[:T,:T]==0, float('-inf'))
        w = w - w.max(-1, keepdim=True)[0]
        return self.wo((torch.nn.functional.softmax(w, dim=-1) @ v).transpose(1,2).reshape(B,T,-1))

class FFN(torch.nn.Module):
    def __init__(self, dim, h):
        super().__init__()
        self.w1 = torch.nn.Linear(dim, h, False)
        self.w2 = torch.nn.Linear(h, dim, False)
        self.w3 = torch.nn.Linear(dim, h, False)
    def forward(self, x): return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))

class Block(torch.nn.Module):
    def __init__(self, dim, nh, nkv, ffn, ckpt=True):
        super().__init__()
        self.ln1 = RMSNorm(dim); self.ln2 = RMSNorm(dim)
        self.attn = GQA(dim, nh, nkv); self.ffn = FFN(dim, ffn)
        self.ckpt = ckpt
    def forward(self, x):
        if self.ckpt and self.training:
            x = x + torch.utils.checkpoint.checkpoint(self._attn_pass, self.ln1(x))
            x = x + torch.utils.checkpoint.checkpoint(self._ffn_pass, self.ln2(x))
        else:
            x = x + self.attn(self.ln1(x)); x = x + self.ffn(self.ln2(x))
        return x
    def _attn_pass(self, x): return self.attn(x)
    def _ffn_pass(self, x): return self.ffn(x)

class LM(torch.nn.Module):
    def __init__(self, V, dim, layers, heads, ffn, ckpt=True):
        super().__init__()
        self.tok = torch.nn.Embedding(V, dim)
        self.blocks = torch.nn.ModuleList([Block(dim, heads, max(2, heads//4), ffn, ckpt) for _ in range(layers)])
        self.ln_f = RMSNorm(dim); self.out = torch.nn.Linear(dim, V, False)
    def forward(self, x):
        x = self.tok(x)
        for b in self.blocks: x = b(x)
        return self.out(self.ln_f(x))

print(f"Loading model...", flush=True)
model = LM(V, 1024, 24, 16, 3072, ckpt=True).to(device)
state = torch.load(r'C:\Users\neura\lm300m_v2_step70000.pt', map_location=device, weights_only=True)
model.load_state_dict(state)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01, betas=(0.9, 0.95))

print(f"GPU mem before: {torch.cuda.memory_allocated()/1024**3:.2f} GB", flush=True)

# Test: 8 gradient accumulation steps
x = torch.randint(0, V, (1, 255)).to(device)
y = torch.randint(0, V, (1, 255)).to(device)

t0 = time.time()
for i in range(8):
    loss = torch.nn.functional.cross_entropy(model(x).view(-1, V), y.view(-1))
    (loss/8).backward()
t1 = time.time()
print(f"8 micro-batches: {t1-t0:.3f}s", flush=True)
print(f"  {(t1-t0)/8*1000:.1f}ms each", flush=True)

opt.step()
opt.zero_grad()
t2 = time.time()
print(f"Optimizer step: {t2-t1:.3f}s", flush=True)

print(f"GPU mem after: {torch.cuda.memory_allocated()/1024**3:.2f} GB", flush=True)
print("TEST OK", flush=True)
