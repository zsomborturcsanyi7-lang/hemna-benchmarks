"""Profiling: find the bottleneck"""
import torch, time, glob, os
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load(r'C:\NeuraNode\bitnet\data\bitnet_pretrain\tokenizer\tokenizer.model')
V = sp.GetPieceSize()
device = 'cuda'

class RMSNorm(torch.nn.Module):
    def __init__(self, dim): super().__init__(); self.w = torch.nn.Parameter(torch.ones(dim))
    def forward(self, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.w

class GQA(torch.nn.Module):
    def __init__(self, dim, nh, nkv):
        super().__init__()
        self.nh, self.nkv, self.hd, self.nr = nh, nkv, dim//nh, nh//nkv
        self.wq = torch.nn.Linear(dim, nh*self.hd, False); self.wk = torch.nn.Linear(dim, nkv*self.hd, False)
        self.wv = torch.nn.Linear(dim, nkv*self.hd, False); self.wo = torch.nn.Linear(nh*self.hd, dim, False)
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
        self.w1 = torch.nn.Linear(dim, h, False); self.w2 = torch.nn.Linear(h, dim, False); self.w3 = torch.nn.Linear(dim, h, False)
    def forward(self, x): return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))

class Block(torch.nn.Module):
    def __init__(self, dim, nh, nkv, ffn):
        super().__init__()
        self.ln1 = RMSNorm(dim); self.ln2 = RMSNorm(dim)
        self.attn = GQA(dim, nh, nkv); self.ffn = FFN(dim, ffn)
    def forward(self, x):
        x = x + torch.utils.checkpoint.checkpoint(self._attn_pass, self.ln1(x), use_reentrant=False)
        x = x + torch.utils.checkpoint.checkpoint(self._ffn_pass, self.ln2(x), use_reentrant=False)
        return x
    def _attn_pass(self, x): return self.attn(x)
    def _ffn_pass(self, x): return self.ffn(x)

class LM(torch.nn.Module):
    def __init__(self, V, dim, layers, heads, ffn):
        super().__init__()
        self.tok = torch.nn.Embedding(V, dim)
        self.blocks = torch.nn.ModuleList([Block(dim, heads, max(2, heads//4), ffn) for _ in range(layers)])
        self.ln_f = RMSNorm(dim); self.out = torch.nn.Linear(dim, V, False)
    def forward(self, x):
        x = self.tok(x)
        for b in self.blocks: x = b(x)
        return self.out(self.ln_f(x))

print("Loading model...", flush=True)
model = LM(V, 1024, 24, 16, 3072).to(device)
state = torch.load(r'C:\Users\neura\lm300m_v2_step70000.pt', map_location=device, weights_only=True)
model.load_state_dict(state)
opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
print(f"Model loaded. GPU mem: {torch.cuda.memory_allocated()/1024**3:.2f}GB", flush=True)

x = torch.randint(0, V, (1, 255)).to(device)
y = torch.randint(0, V, (1, 255)).to(device)

# Detailed profiling of ONE step
print("\n=== DETAILED PROFILE ===", flush=True)

for step in range(5):
    # GPU warmup before measurement  
    a = torch.randn(3000,3000,device=device); b = torch.randn(3000,3000,device=device); _ = a@b
    torch.cuda.synchronize()
    
    t_start = time.time()
    
    t0 = time.time()
    out = model(x)
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Step {step}: forward={t1-t0:.4f}s", flush=True)
    
    t0 = time.time()
    loss = torch.nn.functional.cross_entropy(out.view(-1, V), y.view(-1))
    t1 = time.time()
    print(f"         loss={t1-t0:.4f}s", flush=True)
    
    t0 = time.time()
    loss.backward()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"         backward={t1-t0:.4f}s", flush=True)
    
    t0 = time.time()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"         clip_grad={t1-t0:.4f}s", flush=True)
    
    t0 = time.time()
    opt.step()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"         opt.step={t1-t0:.4f}s", flush=True)
    
    t0 = time.time()
    opt.zero_grad()
    t1 = time.time()
    print(f"         zero_grad={t1-t0:.4f}s", flush=True)
    
    t_end = time.time()
    print(f"         TOTAL={t_end-t_start:.4f}s", flush=True)

print("\nDONE", flush=True)
