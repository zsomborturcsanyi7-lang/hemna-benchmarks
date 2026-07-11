"""Quick test: batch=1 vs batch=2 vs batch=4 speed"""
import torch, os, time
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.set_num_threads(1)
device = 'cuda'
V = 32000; dim = 1024

class RMSNorm(torch.nn.Module):
    def __init__(self, dim): super().__init__(); self.w = torch.nn.Parameter(torch.ones(dim))
    def forward(self, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.w

class GQA(torch.nn.Module):
    def __init__(self, dim, nh, nkv):
        super().__init__()
        self.nh, self.nkv, self.hd = nh, nkv, dim//nh
        self.wq = torch.nn.Linear(dim, nh*self.hd, False); self.wk = torch.nn.Linear(dim, nkv*self.hd, False)
        self.wv = torch.nn.Linear(dim, nkv*self.hd, False); self.wo = torch.nn.Linear(nh*self.hd, dim, False)
        self.register_buffer('m', torch.tril(torch.ones(512, 512)))
    def forward(self, x):
        B,T = x.shape[:2]
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
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class LM(torch.nn.Module):
    def __init__(self): super().__init__()
    def build(self, V, dim, layers, heads, ffn):
        self.tok = torch.nn.Embedding(V, dim)
        self.blocks = torch.nn.ModuleList([Block(dim, heads, max(2, heads//4), ffn) for _ in range(layers)])
        self.ln_f = RMSNorm(dim); self.out = torch.nn.Linear(dim, V, False)
    def forward(self, x):
        x = self.tok(x)
        for b in self.blocks: x = b(x)
        return self.out(self.ln_f(x))

torch.cuda.empty_cache()
print("Loading model...")
model = LM().to(device)
model.build(V, dim, 24, 16, 3072)
ckpt = torch.load(r'C:\Users\neura\lm300m_v2_step285000.pt', map_location='cpu', weights_only=True)
model.load_state_dict(ckpt)
del ckpt
model = model.to(device)
torch.cuda.empty_cache()
print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.1f}/{torch.cuda.memory_reserved()/1024**3:.1f}GB")

# Test data
x1 = torch.randint(0, V, (1, 255), device=device)
y1 = torch.randint(0, V, (1, 255), device=device)
x2 = torch.randint(0, V, (2, 255), device=device)
y2 = torch.randint(0, V, (2, 255), device=device)
x4 = torch.randint(0, V, (4, 255), device=device)
y4 = torch.randint(0, V, (4, 255), device=device)

N = 100  # steps for timing

for bs, x, y in [(1, x1, y1), (2, x2, y2), (4, x4, y4)]:
    torch.cuda.empty_cache()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01, betas=(0.9, 0.95))
    
    # warmup
    for _ in range(3):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x).view(-1, V), y.view(-1))
        loss.backward()
        opt.step()
    
    torch.cuda.synchronize()
    s = time.time()
    for i in range(N):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x).view(-1, V), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    torch.cuda.synchronize()
    dt = time.time() - s
    tok_s = N * bs * 255 / dt
    mem_a = torch.cuda.memory_allocated() / 1024**3
    mem_r = torch.cuda.memory_reserved() / 1024**3
    print(f"batch={bs}: {N} steps in {dt:.2f}s = {dt/N:.4f}s/step = {tok_s:.0f} tok/s  VRAM={mem_a:.1f}/{mem_r:.1f}GB")
