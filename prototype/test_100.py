"""300M modell 100 tesztje"""
import torch, time, codecs
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load(r'C:\NeuraNode\bitnet\data\bitnet_pretrain\tokenizer\tokenizer.model')
V = sp.GetPieceSize()
device = 'cuda'

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
        self.w1 = torch.nn.Linear(dim, h, False); self.w2 = torch.nn.Linear(h, dim, False); self.w3 = torch.nn.Linear(dim, h, False)
    def forward(self, x): return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))

class Block(torch.nn.Module):
    def __init__(self, dim, nh, nkv, ffn):
        super().__init__()
        self.ln1 = RMSNorm(dim); self.ln2 = RMSNorm(dim)
        self.attn = GQA(dim, nh, nkv); self.ffn = FFN(dim, ffn)
    def forward(self, x):
        x = x + self.attn(self.ln1(x)); x = x + self.ffn(self.ln2(x)); return x

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
    def generate(self, prompt_ids, n=80, t=0.7, k=50):
        self.eval()
        x = prompt_ids.unsqueeze(0).to(device)
        out = prompt_ids.tolist()
        for _ in range(n):
            l = self(x[:,-256:])[0,-1]
            if k > 0: v,_ = torch.topk(l, k); l[l < v[-1]] = float('-inf')
            out.append(torch.multinomial(torch.nn.functional.softmax(l/t, dim=-1), 1).item())
            x = torch.cat([x, torch.tensor([[out[-1]]], device=device)], dim=1)
        return sp.DecodeIds(out)

print("Loading 300M model...")
model = LM(V, 1024, 24, 16, 3072).to(device)
model.load_state_dict(torch.load(r'C:\Users\neura\lm300m_final.pt', map_location=device, weights_only=True))
p = sum(p.numel() for p in model.parameters())
print(f"Params: {p:,}")

# 100 prompt - minden tema
prompts = [
    # Beszelgetes / chat
    "Szia! Hogy vagy?",
    "Mi a neved?",
    "Hol laksz?",
    "Meselj magadrol!",
    "Hany eves vagy?",
    "Szeretsz tanulni?",
    "Mi a hobbyid?",
    "Van baratnod?",
    "Mit csinaltal ma?",
    "Unatkozom, mit csinaljunk?",
    
    # MI / technologia
    "Mi az a mesterseges intelligencia?",
    "Hogyan mukodik a gepi tanulas?",
    "Mi a kulonbseg az AI es a ML kozott?",
    "Mire hasznalhato a ChatGPT?",
    "Mi az a neuralis halozat?",
    "Hogyan programozz Pythonban?",
    "Mi a legjobb programozasi nyelv?",
    "Hogyan keszul egy weboldal?",
    "Mi az a blockchain?",
    "Mit jelent a felho?",
    
    # Tudomany
    "Miert kek az eg?",
    "Hogyan mukodik a gravitacio?",
    "Mi az a fekete lyuk?",
    "Van elet a Marson?",
    "Mibol all a DNS?",
    "Hogyan mukodik az emberi agy?",
    "Mi a kulonbseg a DNS es az RNS kozott?",
    "Hogyan keletkezett a Hold?",
    "Mi az a globalis felmelegedes?",
    "Miert van szuksegunk vitaminokra?",
    
    # Magyarorszag / tortenelem
    "Ki volt Szent Istvan?",
    "Mit tudsz Magyarorszagrol?",
    "Mikor volt a mohacsi csata?",
    "Ki volt Petofi Sandor?",
    "Mi tortent 1956-ban?",
    "Mi a magyar konyha specialitasa?",
    "Melyek Magyarorszag szomszedai?",
    "Ki irta a Toldit?",
    "Mi az a Sziget Fesztival?",
    "Hol talalhato a Balaton?",
    
    # Sport
    "Ki nyerte a 2022-es foci VB-t?",
    "Mi a kedvenc sportod?",
    "Ki a legjobb magyar uszo?",
    "Hogyan jatszodik a sakk?",
    "Mi az a Formula 1?",
    "Ki a legertekesebb focista?",
    
    # Gasztro
    "Hogyan keszul a gulyas?",
    "Mi a legjobb magyar etel?",
    "Hogyan sutunk kenyeret?",
    "Mi a kulonbseg a tea es a kave kozott?",
    "Mi az a palinka?",
    
    # Mindennapok
    "Mit egyunk vacsorara?",
    "Hogyan tanulj meg angolul?",
    "Mi a legjobb konyv, amit olvastal?",
    "Hogyan sporoljunk penzt?",
    "Milyen allatot tartsak?",
    "Hogyan fosunk fogat?",
    "Mi a jobb: macska vagy kutya?",
    "Hogyan uljunk repulore?",
    "Mit csinaljunk, ha unatkozunk?",
    "Mennyi alvasra van szuksegem?",
    
    # Film / szorakozas
    "Milyen filmet nezzek?",
    "Ki a kedvenc szineszed?",
    "Mi a legjobb sorozat?",
    "Milyen zenet hallgassak?",
    "Ki a legjobb magyar szinesz?",
    
    # Erzelmek / filozofia
    "Mi az elet ertelme?",
    "Mit jelent a boldogsag?",
    "Mi a szeretet?",
    "Mi a halal?",
    "Van isten?",
    "Mi a szabadsag?",
    "Mi a baratsag?",
    "Mi a penz?",
    "Mi a cel az eletben?",
    "Mit jelent ferfinak/noinek lenni?",
    
    # Altalanos kerdesek
    "Meselj egy viccet!",
    "Mi a legnagyobb talalmany?",
    "Hogyan keszul a papir?",
    "Miert alszunk?",
    "Hogyan mukodik az internet?",
    "Mi a legtobb ember a Foldon?",
    "Melyik a legnagyobb orszag?",
    "Mi a leggyorsabb allat?",
    "Milyen mely az ocean?",
    "Mi a raketatudomany?",
    
    # Random hungaricum
    "Mi az a langos?",
    "Mi a kulonbseg a diszno es a malac kozott?",
    "Mi az a kurtoskalacs?",
    "Hogyan keszul a bejgli?",
    "Mi az a szalonnasutes?",
    "Mit jelent a 'Nemzeti dohanybolt'?",
    "Mi a Tuja?",
    "Ki volt Hofi Geza?",
    "Mi az a Dunakanyar?",
    "Mi a legszebb magyar falu?",
    
    # Zart kerdesek
    "Igen vagy nem?",
    "Melyiket valasztanad?",
]

print(f"\nTeszteles: {len(prompts)} prompt...")
out_file = r'C:\NeuraNode\hemna_bench\test_300m_100.txt'

with codecs.open(out_file, 'w', 'utf-8') as f:
    f.write(f"300M MODELL - 100 TESZT\n")
    f.write(f"Parameterek: {p:,}\n")
    f.write(f"Lepesek: 50K / Adat: 2.64B token\n")
    f.write("="*70 + "\n\n")
    
    magyar = 0
    ertelmes = 0
    dialog = 0
    s = time.time()
    
    for i, prompt in enumerate(prompts):
        ids = torch.tensor(sp.EncodeAsIds(prompt))
        text = model.generate(ids, n=100, t=0.7, k=50)
        
        f.write(f"{i+1}. [{prompt}]\n")
        f.write(f"   {text[:250]}\n\n")
        
        # Egyszeru osztalyozas
        van_magyar = any(c in 'aáeéiíoóöőuúüű' for c in text[:50])
        van_ertelmes = len(text.split()) > 3
        van_parbeszed = any(w in text.lower() for w in ['?', ' -', '- ', 'mondta', 'kerdezte', 'valaszol'])
        
        if van_magyar: magyar += 1
        if van_ertelmes: ertelmes += 1
        if van_parbeszed: dialog += 1
        
        if (i+1) % 20 == 0:
            print(f"  {i+1}/{len(prompts)}")
    
    elapsed = time.time() - s
    f.write("="*70 + "\n")
    f.write(f"STATISZTIKA:\n")
    f.write(f"  Osszes: {len(prompts)}\n")
    f.write(f"  Magyar karakteres valasz: {magyar}/{len(prompts)} ({magyar/len(prompts)*100:.0f}%)\n")
    f.write(f"  Tobb mint 3 szo: {ertelmes}/{len(prompts)} ({ertelmes/len(prompts)*100:.0f}%)\n")
    f.write(f"  Parbeszed jellegu: {dialog}/{len(prompts)} ({dialog/len(prompts)*100:.0f}%)\n")
    f.write(f"  Idotartam: {elapsed:.0f}s ({(elapsed/len(prompts)):.1f}s / prompt)\n")

print(f"\nKesz! Eredmenyek: {out_file}")
print(f"Stat: magyar={magyar}, ertelmes={ertelmes}, parbeszed={dialog}")
