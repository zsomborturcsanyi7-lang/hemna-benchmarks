"""Lassulas okat keressuk"""
import torch, time
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda'
dim, nh, nkv, hd = 1024, 16, 4, 64
B, T = 2, 256

class GQA(nn.Module):
    def __init__(self):
        super().__init__()
        self.nh, self.nkv, self.hd, self.nr = nh, nkv, hd, nh//nkv
        self.wq = nn.Linear(dim, nh*hd, False)
        self.wk = nn.Linear(dim, nkv*hd, False)
        self.wv = nn.Linear(dim, nkv*hd, False)
        self.wo = nn.Linear(nh*hd, dim, False)
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
        return self.wo((F.softmax(w - w.max(-1,keepdim=True)[0], dim=-1) @ v).transpose(1,2).reshape(B,T,-1))

attn = GQA().to(device)
x = torch.randn(B, T, dim, device=device)

# 1. Norm modul + checkpoint modul
class Norm(nn.Module):
    def __init__(self): super().__init__(); self.w = nn.Parameter(torch.ones(dim))
    def forward(self, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.w

ln = Norm().to(device)

# Teszt 1: modul method checkpoint
def _attn_pass(x): return attn(x)

print("Teszt 1: method checkpoint...")
for _ in range(3):  # warmup
    _ = torch.utils.checkpoint.checkpoint(_attn_pass, ln(x))
s = time.time()
for _ in range(100):
    _ = torch.utils.checkpoint.checkpoint(_attn_pass, ln(x))
t1 = time.time() - s
print(f"  Method: {t1/100*1000:.2f}ms")

# Teszt 2: modul direkt checkpoint
print("Teszt 2: modul direkt checkpoint...")
for _ in range(3):
    _ = torch.utils.checkpoint.checkpoint(attn, ln(x))
s = time.time()
for _ in range(100):
    _ = torch.utils.checkpoint.checkpoint(attn, ln(x))
t2 = time.time() - s
print(f"  Direkt: {t2/100*1000:.2f}ms")

# Teszt 3: use_reentrant parameter
print("Teszt 3: use_reentrant=False...")
for _ in range(3):
    _ = torch.utils.checkpoint.checkpoint(_attn_pass, ln(x), use_reentrant=False)
s = time.time()
for _ in range(100):
    _ = torch.utils.checkpoint.checkpoint(_attn_pass, ln(x), use_reentrant=False)
t3 = time.time() - s
print(f"  use_reentrant=False: {t3/100*1000:.2f}ms")

# Teszt 4: nincs checkpoint (full forward)
print("Teszt 4: nincs checkpoint...")
for _ in range(3):
    _ = attn(ln(x))
s = time.time()
for _ in range(100):
    _ = attn(ln(x))
t4 = time.time() - s
print(f"  Nincs checkpoint: {t4/100*1000:.2f}ms")

# Teszt 5: Teljes model forward (24 layer)
class Block(nn.Module):
    def __init__(self, ckpt_method='method'):
        super().__init__()
        self.ln1 = Norm(); self.ln2 = Norm()
        self.attn = GQA(); self.ffn = nn.Sequential(nn.Linear(1024, 3072, False), nn.SiLU(), nn.Linear(3072, 1024, False))
        self.ckpt_method = ckpt_method
    def forward(self, x):
        if self.training:
            if self.ckpt_method == 'method':
                x = x + torch.utils.checkpoint.checkpoint(self._ap, self.ln1(x))
                x = x + torch.utils.checkpoint.checkpoint(self._fp, self.ln2(x))
            elif self.ckpt_method == 'direct':
                x = x + torch.utils.checkpoint.checkpoint(self.attn, self.ln1(x))
                x = x + torch.utils.checkpoint.checkpoint(self.ffn, self.ln2(x))
        else:
            x = x + self.attn(self.ln1(x)); x = x + self.ffn(self.ln2(x))
        return x
    def _ap(self, x): return self.attn(x)
    def _fp(self, x): return self.ffn(x)

# Teszt 5a: method - 24 layer
m = nn.Sequential(*[Block('method') for _ in range(24)]).to(device).train()
print(f"\nTeszt 5a: 24 layer method checkpoint...")
for _ in range(3): m(x)
s = time.time()
for _ in range(10): m(x)
t5a = time.time() - s
print(f"  {t5a/10*1000:.1f}ms per forward ({2*255*1000/(t5a/10*1000):.0f} tok/s)")

# Teszt 5b: direct - 24 layer
m2 = nn.Sequential(*[Block('direct') for _ in range(24)]).to(device).train()
print(f"Teszt 5b: 24 layer direct checkpoint...")
for _ in range(3): m2(x)
s = time.time()
for _ in range(10): m2(x)
t5b = time.time() - s
print(f"  {t5b/10*1000:.1f}ms per forward ({2*255*1000/(t5b/10*1000):.0f} tok/s)")

# Teszt 5c: nincs checkpoint - 24 layer
m3 = nn.Sequential(*[Block('') for _ in range(24)]).to(device).train()
# Kapcsoljuk ki a checkpointot
for b in m3: b.ckpt_method = 'none'
print(f"Teszt 5c: 24 layer nincs checkpoint...")
for _ in range(3): m3(x)
s = time.time()
for _ in range(10): m3(x)
t5c = time.time() - s
print(f"  {t5c/10*1000:.1f}ms per forward ({2*255*1000/(t5c/10*1000):.0f} tok/s)")

print(f"\nOsszefoglalo:")
print(f"  Method ckpt: {t1/100*1000:.2f}ms")
print(f"  Direct ckpt: {t2/100*1000:.2f}ms")
print(f"  use_reentrant=False: {t3/100*1000:.2f}ms")
print(f"  Nincs ckpt: {t4/100*1000:.2f}ms")
print(f"  24L method: {t5a/10*1000:.1f}ms ({2*255*1000/(t5a/10*1000):.0f} tok/s)")
print(f"  24L direct: {t5b/10*1000:.1f}ms ({2*255*1000/(t5b/10*1000):.0f} tok/s)")
print(f"  24L no ckpt: {t5c/10*1000:.1f}ms ({2*255*1000/(t5c/10*1000):.0f} tok/s)")
