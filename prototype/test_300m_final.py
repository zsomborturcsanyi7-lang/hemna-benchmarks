"""300M vegso modell teszt"""
import torch, os, codecs
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load(r'C:\NeuraNode\bitnet\data\bitnet_pretrain\tokenizer\tokenizer.model')
V = sp.GetPieceSize()

device = 'cuda'
print(f"Modell betoltese...")

# Model definition
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
        if self.nr > 1:
            k = k[:,:,None].expand(-1,-1,self.nr,-1,-1).reshape(B,self.nh,T,self.hd)
            v = v[:,:,None].expand(-1,-1,self.nr,-1,-1).reshape(B,self.nh,T,self.hd)
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
        self.ln_f = RMSNorm(dim)
        self.out = torch.nn.Linear(dim, V, False)
    def forward(self, x):
        x = self.tok(x)
        for b in self.blocks: x = b(x)
        return self.out(self.ln_f(x))
    @torch.no_grad()
    def generate(self, prompt, n=200, t=0.7, k=50):
        self.eval()
        x = prompt.unsqueeze(0).to(device)
        out = prompt.tolist()
        for _ in range(n):
            l = self(x[:,-256:])[0,-1]
            if k > 0: v,_ = torch.topk(l, k); l[l < v[-1]] = float('-inf')
            out.append(torch.multinomial(torch.nn.functional.softmax(l/t, dim=-1), 1).item())
            x = torch.cat([x, torch.tensor([[out[-1]]], device=device)], dim=1)
        return sp.DecodeIds(out)

model = LM(V, 1024, 24, 16, 3072).to(device)
state = torch.load(r'C:\Users\neura\lm300m_final.pt', map_location=device, weights_only=True)
model.load_state_dict(state)
p = sum(p.numel() for p in model.parameters())
print(f"Parameterek: {p:,}")

prompts = [
    "A mesterseges intelligencia",
    "Szia! Hogy vagy?",
    "Mi a veleményed a mai világrol?",
]

# Fileba mentes a kodolasi hiba miatt
out_file = r'C:\NeuraNode\hemna_bench\test_300m_final.txt'
with codecs.open(out_file, 'w', 'utf-8') as f:
    for prompt in prompts:
        ids = torch.tensor(sp.EncodeAsIds(prompt))
        text = model.generate(ids, n=200, t=0.7, k=50)
        f.write(f"\n--- Prompt: {prompt} ---\n")
        f.write(f"Valasz: {text[:300]}\n")

print(f"\nMentve: {out_file}")
print("KESZ")
