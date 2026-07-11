"""300M magyar LM interaktiv teszt CLI - KV cache"""
import torch, os, time, math, glob, re, sys
import sentencepiece as spm
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
if sys.platform == 'win32' or sys.stdout.encoding.lower() in ('cp1250', 'cp1252', 'latin-1'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout.reconfigure(errors='replace')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

sp = spm.SentencePieceProcessor()
sp.Load(r'C:\NeuraNode\bitnet\data\bitnet_pretrain\tokenizer\tokenizer.model')
V = sp.GetPieceSize()

class RMSNorm(torch.nn.Module):
    def __init__(self, dim): super().__init__(); self.w = torch.nn.Parameter(torch.ones(dim))
    def forward(self, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.w

class GQA(torch.nn.Module):
    def __init__(self, dim, nh, nkv):
        super().__init__()
        self.nh, self.nkv, self.hd = nh, nkv, dim//nh
        self.wq = torch.nn.Linear(dim, nh*self.hd, False)
        self.wk = torch.nn.Linear(dim, nkv*self.hd, False)
        self.wv = torch.nn.Linear(dim, nkv*self.hd, False)
        self.wo = torch.nn.Linear(nh*self.hd, dim, False)
    
    def forward(self, x, past_k=None, past_v=None):
        B, T = x.shape[:2]
        q = self.wq(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        
        # KV cache: konkatenaljuk az elozo lepes K,V-hez (nkv-s fejjel)
        cache_k = torch.cat([past_k, k], dim=-2) if past_k is not None else k
        cache_v = torch.cat([past_v, v], dim=-2) if past_v is not None else v
        has_cache = past_k is not None
        
        # GQA: K,V-t kiterjesztjuk nh fejre
        ratio = self.nh // self.nkv
        k_full = cache_k[:, :, None].expand(-1, -1, ratio, -1, -1).reshape(B, self.nh, -1, self.hd)
        v_full = cache_v[:, :, None].expand(-1, -1, ratio, -1, -1).reshape(B, self.nh, -1, self.hd)
        
        # Flash Attention! (beepitett, CUDA optimalizalt)
        is_causal = not has_cache
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k_full, v_full,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=self.hd ** -0.5
        )
        
        out = self.wo(out.transpose(1, 2).reshape(B, T, -1))
        return out, cache_k, cache_v


class FFN(torch.nn.Module):
    def __init__(self, dim, h):
        super().__init__()
        self.w1 = torch.nn.Linear(dim, h, False)
        self.w2 = torch.nn.Linear(h, dim, False)
        self.w3 = torch.nn.Linear(dim, h, False)
    def forward(self, x): return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))


class Block(torch.nn.Module):
    def __init__(self, dim, nh, nkv, ffn):
        super().__init__()
        self.ln1 = RMSNorm(dim); self.ln2 = RMSNorm(dim)
        self.attn = GQA(dim, nh, nkv); self.ffn = FFN(dim, ffn)
    
    def forward(self, x, past_k=None, past_v=None):
        residual = x
        h = self.ln1(x)
        attn_out, k, v = self.attn(h, past_k, past_v)
        x = residual + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, k, v


class LM(torch.nn.Module):
    def __init__(self, V, dim, layers, heads, ffn):
        super().__init__()
        self.tok = torch.nn.Embedding(V, dim)
        self.blocks = torch.nn.ModuleList([Block(dim, heads, max(2, heads//4), ffn) for _ in range(layers)])
        self.ln_f = RMSNorm(dim); self.out = torch.nn.Linear(dim, V, False)
    
    def forward(self, x, kv_cache=None):
        """Normal forward (training). kv_cache=None."""
        h = self.tok(x)
        new_cache = [] if kv_cache is not None else None
        for i, block in enumerate(self.blocks):
            pk = pv = None
            if kv_cache is not None and i < len(kv_cache) and kv_cache[i] is not None:
                pk, pv = kv_cache[i]
            h, k, v = block(h, pk, pv)
            if new_cache is not None:
                new_cache.append((k, v))
        h = self.ln_f(h)
        logits = self.out(h)
        if new_cache is not None:
            return logits, new_cache
        return logits
    
    @torch.no_grad()
    def generate(self, prompt_ids, n_tokens=100, temp=0.7, top_k=50):
        self.eval()
        x = prompt_ids.unsqueeze(0).to(device)
        out = prompt_ids.tolist()
        
        # PREFILL: teljes prompt forward, KV cache letrehozasa
        logits, kv_cache = self(x, kv_cache=[None] * len(self.blocks))
        next_logits = logits[0, -1]
        if top_k > 0:
            vals, _ = torch.topk(next_logits, top_k)
            next_logits[next_logits < vals[-1]] = float('-inf')
        next_id = torch.multinomial(torch.nn.functional.softmax(next_logits / temp, dim=-1), 1).item()
        out.append(next_id)
        
        # DECODE: csak 1 token forward, KV cache-bol
        x = torch.tensor([[next_id]], device=device)
        for _ in range(n_tokens - 1):
            logits, kv_cache = self(x, kv_cache=kv_cache)
            next_logits = logits[0, -1]
            if top_k > 0:
                vals, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < vals[-1]] = float('-inf')
            next_id = torch.multinomial(torch.nn.functional.softmax(next_logits / temp, dim=-1), 1).item()
            out.append(next_id)
            x = torch.tensor([[next_id]], device=device)
        
        return out


# ====== MODELL BETOLTESE ======
print(f"\n{'='*50}")
print(f"  300M magyar LM - Interaktiv Teszt (KV cache)")
print(f"{'='*50}")
print(f"  Device: {device}")
print(f"  Vocab: {V}")
print(f"{'='*50}\n")

print("  Legfrissebb checkpoint keresese...")
ckpt_dir = r'C:\Users\neura'
saved = sorted(glob.glob(os.path.join(ckpt_dir, 'lm300m_v2_step*.pt')))
if not saved:
    base = os.path.join(ckpt_dir, 'lm300m_final.pt')
    if os.path.exists(base):
        saved = [base]
        step = 50000
    else:
        print("  HIBA: nincs checkpoint!")
        sys.exit(1)

latest = saved[-1]
m = re.search(r'step(\d+)', latest)
step = int(m.group(1)) if m else 50000
size_gb = os.path.getsize(latest) / 1024**3

print(f"  Checkpoint: {os.path.basename(latest)}")
print(f"  Meret: {size_gb:.2f} GB")
print(f"  Lepes: {step}K")
print(f"\n  Modell betoltese...", end=' ', flush=True)

model = LM(V, 1024, 24, 16, 3072).to(device)
state = torch.load(latest, map_location=device, weights_only=True)
model.load_state_dict(state, strict=False)
params = sum(p.numel() for p in model.parameters())
print(f"[KESZ]")
print(f"  Parameterek: {params:,}")
if device == 'cuda':
    print(f"  VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
print()

# ====== GYORS TESZTEK ======
print("  Gyors tesztek:")
test_prompts = [
    "Szia! Hogy vagy?",
    "Mi a velemenyed a mesterseges intelligenciarol?",
    "Meselj magadrol!",
    "Hogyan tanultal meg magyarul?",
]

for prompt in test_prompts:
    print(f"\n  -- Prompt: \"{prompt}\"")
    ids = torch.tensor(sp.EncodeAsIds(prompt), dtype=torch.long)
    t0 = time.time()
    out_ids = model.generate(ids, n_tokens=80, temp=0.7, top_k=50)
    text = sp.DecodeIds(out_ids)
    t = time.time() - t0
    tok_s = (len(out_ids) - len(ids)) / t
    print(f"  -- Valasz ({t:.1f}s, {tok_s:.0f} tok/s):")
    print(f"  >> {text}")
    print()

# ====== PPL ======
print(f"\n{'='*50}")
print("  Perplexity meres (OpenSubtitles / HunSum-2)")
print(f"{'='*50}")

DATA_DIR = r'C:\NeuraNode\hemna_bench\combined_no_wiki'
files = sorted(glob.glob(os.path.join(DATA_DIR, '*.pt')))
all_data = torch.cat([torch.load(f, map_location='cpu', weights_only=True) for f in files[:10]], dim=0)
val_data = all_data[-500:]

losses = []
model.eval()
with torch.no_grad():
    for i in range(min(200, len(val_data))):
        seq = val_data[i].to(device, dtype=torch.long)
        x, y = seq[:255].unsqueeze(0), seq[1:256].unsqueeze(0)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, V), y.view(-1))
        losses.append(loss.item())

avg_loss = sum(losses) / len(losses)
ppl = math.exp(avg_loss)
print(f"  Szekvenciak: {len(losses)}")
print(f"  Atlag loss: {avg_loss:.4f}")
print(f"  Perplexity: {ppl:.1f}")
print(f"\n{'='*50}")
print(f"  PPL osszehasonlitas:")
print(f"    lm300m_final.pt (50K):  133.9")
print(f"    Jelenleg ({step}K):     {ppl:.1f}")
print(f"{'='*50}\n")

# ====== INTERAKTIV ======
print("  Interaktiv mod:")
print("  'exit' kilepeshez, 'temp=0.5' a homerseklethez")
print()

while True:
    try:
        line = input("  Prompt> ").strip()
    except EOFError:
        break
    if not line: continue
    if line.lower() in ['exit', 'quit', 'q', '']: break
    if line.startswith('temp='):
        temp = float(line.split('=')[1])
        print(f"  Homerseklet: {temp}")
        continue
    
    ids = torch.tensor(sp.EncodeAsIds(line), dtype=torch.long)
    if len(ids) == 0:
        print("  (ures)")
        continue
    
    t0 = time.time()
    out_ids = model.generate(ids, n_tokens=120, temp=0.7, top_k=50)
    text = sp.DecodeIds(out_ids)
    t = time.time() - t0
    tok_s = (len(out_ids) - len(ids)) / t
    print(f"  [{t:.1f}s, {tok_s:.0f} tok/s]: {text}")
    print()
