"""10 lepes teszt timeouttal"""
import torch, time, glob, signal, sys
import sentencepiece as spm
torch.set_num_threads(1)

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

print("Loading...", flush=True)
model = LM(V, 1024, 24, 16, 3072).to(device)
state = torch.load(r'C:\Users\neura\lm300m_v2_step70000.pt', map_location=device, weights_only=True)
model.load_state_dict(state)
opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

x = torch.randint(0, V, (1, 255)).to(device)
y = torch.randint(0, V, (1, 255)).to(device)
print("Starting 10 steps...", flush=True)

for step in range(10):
    t_start = time.time()
    
    model.train()
    
    t0 = time.time()
    out = model(x)
    torch.cuda.synchronize()
    t1 = time.time()
    
    loss = torch.nn.functional.cross_entropy(out.view(-1, V), y.view(-1))
    t2 = time.time()
    
    loss.backward()
    torch.cuda.synchronize()
    t3 = time.time()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    t4 = time.time()
    
    opt.step()
    torch.cuda.synchronize()
    t5 = time.time()
    
    opt.zero_grad()
    t6 = time.time()
    
    print(f"Step {step}: fwd={t1-t0:.3f}s loss={t2-t1:.3f}s bwd={t3-t2:.3f}s clip={t4-t3:.3f}s opt={t5-t4:.3f}s zero={t6-t5:.3f}s tot={t6-t_start:.3f}s", flush=True)
    
    # GPU clock check
    import subprocess
    clk = subprocess.check_output('nvidia-smi --query-gpu=clocks.current.graphics --format=csv,noheader', shell=True).decode().strip()
    print(f"  GPU clock: {clk}", flush=True)

print("DONE", flush=True)
