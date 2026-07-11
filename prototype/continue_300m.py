"""300M folytatas - AMP + batch_size=4 + empty_cache"""
import torch, os, time, math, glob, re
import sentencepiece as spm
from torch.cuda.amp import autocast, GradScaler
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.set_num_threads(1)

sp = spm.SentencePieceProcessor()
sp.Load(r'C:\NeuraNode\bitnet\data\bitnet_pretrain\tokenizer\tokenizer.model')
V = sp.GetPieceSize()

LOG = r'C:\NeuraNode\hemna_bench\continue_300m_log.txt'
def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    print(msg, flush=True)

device = 'cuda'
log(f"Device: {device}")

# ====== ARCHITEKTURA (NO CKPT) ======
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
    @torch.no_grad()
    def generate(self, prompt, n=100, t=0.7, k=50):
        self.eval()
        x = prompt.unsqueeze(0).to(device)
        out = prompt.tolist()
        for _ in range(n):
            l = self(x[:,-256:])[0,-1]
            if k > 0: v,_ = torch.topk(l, k); l[l < v[-1]] = float('-inf')
            out.append(torch.multinomial(torch.nn.functional.softmax(l/t, dim=-1), 1).item())
            x = torch.cat([x, torch.tensor([[out[-1]]], device=device)], dim=1)
        return sp.DecodeIds(out)

# ====== ADAT ======
DATA_DIR = r'C:\NeuraNode\hemna_bench\combined_no_wiki'
files = sorted(glob.glob(os.path.join(DATA_DIR, '*.pt')))
log(f"Adat: {len(files)} shard")
all_data = torch.cat([torch.load(f, map_location='cpu', weights_only=True) for f in files], dim=0)
log(f"  {all_data.shape[0]:,} seq")
train_d = all_data[:int(len(all_data)*0.98)]
val_d = all_data[int(len(all_data)*0.98):]
del all_data

BATCH_SIZE = 4

def get_batch(d, bs=BATCH_SIZE):
    ix = torch.randint(len(d), (bs,))
    x = d[ix, :255].to(device, dtype=torch.long)
    y = d[ix, 1:256].to(device, dtype=torch.long)
    return x, y

# ====== MODELL ======
log("Modell betoltese...")
model = LM(V, 1024, 24, 16, 3072).to(device)
state = torch.load(r'C:\Users\neura\lm300m_final.pt', map_location=device, weights_only=True)
saved = sorted(glob.glob(r'C:\\Users\\neura\\lm300m_v2_step*.pt'), key=lambda f: int(re.search(r'step(\d+)', f).group(1)))
start = 50000
if saved:
    latest = saved[-1]
    log(f"Checkpoint: {os.path.basename(latest)}")
    state = torch.load(latest, map_location=device, weights_only=True)
    m = re.search(r'step(\d+)', latest)
    if m: start = int(m.group(1))
model.load_state_dict(state)
log(f"Params: {sum(p.numel() for p in model.parameters()):,} ({start}. lepes)")

opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01, betas=(0.9, 0.95))
scaler = GradScaler()

STEPS, LOG_EVERY, SAVE_EVERY = 100000, 500, 5000
log(f"Fut: {STEPS} lepes, batch={BATCH_SIZE}, AMP=True, empty_cache=100")

s = time.time()
for step in range(STEPS):
    model.train()
    x, y = get_batch(train_d)
    
    with autocast():
        loss = torch.nn.functional.cross_entropy(model(x).view(-1, V), y.view(-1))
    loss_val = loss.item()
    
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    opt.zero_grad()
    
    if step % 100 == 0 and step % LOG_EVERY != 0:
        torch.cuda.empty_cache()
    
    if step % LOG_EVERY == 0:
        model.eval()
        with torch.no_grad():
            xv, yv = get_batch(val_d, 1)  # val batch=1 is fine
            with autocast():
                vl = torch.nn.functional.cross_entropy(model(xv).view(-1, V), yv.view(-1)).item()
        tok_s = BATCH_SIZE * 255 * LOG_EVERY / (time.time() - s + 1e-8)
        mem_a = torch.cuda.memory_allocated() / 1024**3
        mem_r = torch.cuda.memory_reserved() / 1024**3
        log(f"Step {step:5d}/{STEPS} (total {start+step:5d}): loss={loss_val:.4f} val={vl:.4f} ppl={math.exp(vl):.1f} tok/s={tok_s:.0f} VRAM={mem_a:.1f}/{mem_r:.1f}GB")
        s = time.time()
        torch.cuda.empty_cache()
    
    if step > 0 and step % SAVE_EVERY == 0:
        torch.save(model.state_dict(), f'C:\\Users\\neura\\lm300m_v2_step{start+step}.pt')
        test = torch.tensor(sp.EncodeAsIds("Szia! Hogy vagy?"))
        txt = model.generate(test, 100, 0.7)
        log(f"  Gen: {txt[:150]}")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

torch.save(model.state_dict(), f'C:\\Users\\neura\\lm300m_v2_final.pt')
log("DONE")
