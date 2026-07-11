"""Quick test: batch=1 vs batch=4 speed"""
import torch, os, time
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.set_num_threads(1)
device = 'cuda'

V = 32000
dim = 1024

# Build a single block to test
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
    def __init__(self, V, dim, layers, heads, ffn):
        super().__init__()
        self.tok = torch.nn.Embedding(V, dim)
        self.blocks = torch.nn.ModuleList([Block(dim, heads, max(2, heads//4), ffn) for _ in range(layers)])
        self.ln_f = RMSNorm(dim); self.out = torch.nn.Linear(dim, V, False)
    def forward(self, x):
        x = self.tok(x)
        for b in self.blocks: x = b(x)
        return self.out(self.ln_f(x))

print(f"Building 24-layer model ({sum(p.numel() for p in LM(V, dim, 24, 16, 3072).parameters()):,} params)...")
model = LM(V, dim, 24, 16, 3072).to(device)

# Load actual checkpoint
ckpt = torch.load(r'C:\Users\neura\lm300m_v2_step285000.pt', map_location=device, weights_only=True)
model.load_state_dict(ckpt)
print("Checkpoint loaded.")

x1 = torch.randint(0, V, (1, 255), device=device)
y1 = torch.randint(0, V, (1, 255), device=device)
x4 = torch.randint(0, V, (4, 255), device=device)
y4 = torch.randint(0, V, (4, 255), device=device)

# Warmup
for _ in range(3):
    model(x1[:,-255:])

# Test batch=1
opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01, betas=(0.9, 0.95))
torch.cuda.synchronize()
s = time.time()
for i in range(50):
    opt.zero_grad()
    loss = torch.nn.functional.cross_entropy(model(x1).view(-1, V), y1.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if i % 100 == 0 and i % 500 != 0:
        torch.cuda.empty_cache()
torch.cuda.synchronize()
dt1 = time.time() - s
tok_s_1 = 50 * 255 / dt1
mem_a = torch.cuda.memory_allocated() / 1024**3
mem_r = torch.cuda.memory_reserved() / 1024**3
print(f"batch=1: {50} steps in {dt1:.2f}s = {dt1/50:.4f}s/step = {tok_s_1:.0f} tok/s  VRAM={mem_a:.1f}/{mem_r:.1f}GB")

# Test batch=4
opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01, betas=(0.9, 0.95))
torch.cuda.empty_cache()
torch.cuda.synchronize()
s = time.time()
for i in range(50):
    opt.zero_grad()
    loss = torch.nn.functional.cross_entropy(model(x4).view(-1, V), y4.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if i % 100 == 0 and i % 500 != 0:
        torch.cuda.empty_cache()
torch.cuda.synchronize()
dt4 = time.time() - s
tok_s_4 = 50 * 4 * 255 / dt4
mem_a = torch.cuda.memory_allocated() / 1024**3
mem_r = torch.cuda.memory_reserved() / 1024**3
print(f"batch=4: {50} steps in {dt4:.2f}s = {dt4/50:.4f}s/step = {tok_s_4:.0f} tok/s  VRAM={mem_a:.1f}/{mem_r:.1f}GB")
