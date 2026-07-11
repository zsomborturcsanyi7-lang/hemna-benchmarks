"""HEMNA MicroLM - hasznalhato magyar nyelvu modell"""
import torch, torch.nn as nn, torch.nn.functional as F, sys, time, math, json, os
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MicroLM(nn.Module):
    """Optimalizalt mikro nyelvi modell magyar nyelvre."""
    def __init__(self, vocab_size, dim=256, layers=6, heads=8, ffn=512, window=128, drop=0.1):
        super().__init__()
        self.dim = dim
        self.window = window
        self.tok = nn.Embedding(vocab_size, dim)
        self.drop = nn.Dropout(drop)
        
        head_dim = dim // heads
        nkv = max(2, heads // 4)
        self.heads = heads
        self.nkv = nkv
        self.head_dim = head_dim
        self.nr = heads // nkv
        
        # Layers
        self.wq = nn.ModuleList([nn.Linear(dim, heads*head_dim, False) for _ in range(layers)])
        self.wk = nn.ModuleList([nn.Linear(dim, nkv*head_dim, False) for _ in range(layers)])
        self.wv = nn.ModuleList([nn.Linear(dim, nkv*head_dim, False) for _ in range(layers)])
        self.wo = nn.ModuleList([nn.Linear(heads*head_dim, dim, False) for _ in range(layers)])
        self.ffn_w1 = nn.ModuleList([nn.Linear(dim, ffn, False) for _ in range(layers)])
        self.ffn_w2 = nn.ModuleList([nn.Linear(ffn, dim, False) for _ in range(layers)])
        self.ffn_w3 = nn.ModuleList([nn.Linear(dim, ffn, False) for _ in range(layers)])
        self.ln1 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])
        
        self.ln_f = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, vocab_size, False)  # tied embedding
        self.register_buffer('cmask', torch.tril(torch.ones(1024, 1024)))
    
    def forward(self, x, return_hidden=False):
        B, T = x.shape
        x = self.drop(self.tok(x))
        
        for i in range(len(self.wq)):
            # GQA Attention
            q = self.wq[i](x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
            k = self.wk[i](x).view(B, T, self.nkv, self.head_dim).transpose(1, 2)
            v = self.wv[i](x).view(B, T, self.nkv, self.head_dim).transpose(1, 2)
            
            if self.nr > 1:
                k = k[:,:,None].expand(-1,-1,self.nr,-1,-1).reshape(B,self.heads,T,self.head_dim)
                v = v[:,:,None].expand(-1,-1,self.nr,-1,-1).reshape(B,self.heads,T,self.head_dim)
            
            wei = (q @ k.transpose(-2,-1)) * (self.head_dim**-0.5)
            wei = wei.masked_fill(self.cmask[:T,:T]==0, float('-inf'))
            wei = wei.masked_fill(torch.triu(torch.ones(T,T,device=x.device), self.window+1)==1, float('-inf'))
            wei = F.softmax(wei, dim=-1)
            attn_out = (wei @ v).transpose(1, 2).reshape(B, T, -1)
            
            x = x + self.drop(self.wo[i](attn_out))
            x = x + self.drop(self.ffn_w2[i](F.silu(self.ffn_w1[i](self.ln2[i](x))) * self.ffn_w3[i](self.ln2[i](x))))
        
        x = self.ln_f(x)
        logits = self.out(x)
        return logits
    
    @torch.no_grad()
    def generate(self, prompt, max_len=200, temp=0.7, top_k=20):
        self.eval()
        x = prompt.unsqueeze(0).to(device)
        out = prompt.tolist()
        for _ in range(max_len):
            logits = self(x[:, -self.window:])[0, -1]
            if top_k > 0:
                values, _ = torch.topk(logits, top_k)
                logits[logits < values[-1]] = float('-inf')
            probs = F.softmax(logits / temp, dim=-1)
            nxt = torch.multinomial(probs, 1).item()
            out.append(nxt)
            x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
        return torch.tensor(out)


def train_model(text, dim=256, layers=6, steps=3000, save_path="micro_lm.pt"):
    """Teljes training pipeline."""
    print(f"Adat elokeszites...")
    chars = sorted(list(set(text.lower())))
    V = len(chars)
    c2i = {c:i for i,c in enumerate(chars)}
    i2c = {i:c for i,c in enumerate(chars)}
    data = torch.tensor([c2i[c] for c in text.lower()])
    train_d, val_d = data[:int(len(data)*0.9)], data[int(len(data)*0.9):]
    print(f"  Karakter: {len(text)}, szokincs: {V}")
    
    def get_batch(d, bs=32, bl=128):
        ix = torch.randint(len(d)-bl-1, (bs,))
        x = torch.stack([d[i:i+bl] for i in ix]).to(device)
        y = torch.stack([d[i+1:i+bl+1] for i in ix]).to(device)
        return x, y
    
    model = MicroLM(V, dim, layers).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameterek: {params:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    
    print(f"\nTanitas {steps} lepes...")
    s = time.time()
    for step in range(steps):
        model.train()
        x, y = get_batch(train_d, 32, 128)
        opt.zero_grad()
        loss = F.cross_entropy(model(x).view(-1, V), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        
        if step % 500 == 0:
            model.eval()
            with torch.no_grad():
                xv, yv = get_batch(val_d, 32, 128)
                vl = F.cross_entropy(model(xv).view(-1, V), yv.view(-1)).item()
            ppl = math.exp(vl)
            print(f"  Step {step:4d}: loss={loss.item():.4f}, val_loss={vl:.4f}, ppl={ppl:.2f}, lr={sched.get_last_lr()[0]:.2e}")
    
    model.eval()
    with torch.no_grad():
        losses = []
        for _ in range(20):
            xv, yv = get_batch(val_d, 32, 128)
            losses.append(F.cross_entropy(model(xv).view(-1, V), yv.view(-1)).item())
    final_loss = sum(losses)/len(losses)
    print(f"\nKesz! Loss: {final_loss:.4f}, Perplexity: {math.exp(final_loss):.2f}")
    print(f"Ido: {time.time()-s:.1f}s")
    
    # Mentes
    torch.save({
        'model_state': model.state_dict(),
        'vocab_size': V,
        'dim': dim,
        'layers': layers,
        'c2i': c2i,
        'i2c': i2c,
        'loss': final_loss
    }, save_path)
    print(f"Modell mentve: {save_path}")
    
    return model, c2i, i2c


def chat(model, c2i, i2c, prompt, max_len=200, temp=0.7):
    """Valasz generalasa promptra."""
    prompt = prompt.lower().strip()
    input_ids = torch.tensor([c2i.get(c, 0) for c in prompt])
    output_ids = model.generate(input_ids, max_len, temp)
    return ''.join(i2c[i] for i in output_ids.tolist())


if __name__ == '__main__':
    print(f"Eszkoz: {device}")
    
    # Bovebb magyar szoveg
    corpus = """
az ember agya egy csodalatos szerkezet. kutatasai a mai napig tartanak.
a mesterseges intelligencia olyan terulet ahol a szamitogepeket probaljuk
emberi intelligenciaval felruhazni. a gepi tanulas segitsegevel a szamitogepek
kepesek mintakat felismerni az adatokbol.
a mai idojaras napos es meleg. holnap varnato eso is lehet.
a macska az asztal alatt alszik. a kutya a kertben jatszik.
a matematika hatt feladatot kaptam. meg kell oldanom a harmadik peldat.
a programozas izgalmas hobbi. uj nyelveket tanulni mindig erdekes.
az alma egeszseges. a sport segit. a baratok tamaszt nyujtanak.
az utazas szinesiti az eletet. a technologia folyamatosan fejlodik.
a reggeli a nap legfontosabb etkezese. minden nap edzek egy kicsit.
a konyvek fontosak a tanulashoz. egy jo konyv eletre szolo tudast ad.
a zenet mindenki szereti. a klasszikus zene megnyugtat.
a kave finom ital. a tej egeszseges. a viz az elete.
a nap sut. a hold vilagit. a csillagok szepen ragyognak.
az erdoben allatok elnek. a folyo vizet szallit. a hegyek magasak.
az iskolaban sokat tanulunk. a tanar segit megerteni a vilagot.
a baratommal egyutt jatszunk. a csalad fontos. a szulok szeretnek.
a telefon segit kommunikalni. az internet osszekoti az embereket.
a gep gyorsan szamol. a program szepen fut. a kod tiszta es ertheto.
""".lower().strip()

    # Training
    model, c2i, i2c = train_model(corpus, dim=192, layers=4, steps=2000)
    
    # Interaktiv teszt
    print("\n=== Chat teszt ===")
    test_prompts = [
        "a mesterseges intelligencia",
        "a macska es a kutya",
        "a programozas",
        "az idojaras",
    ]
    for p in test_prompts:
        print(f"\nKerdes: {p}")
        valasz = chat(model, c2i, i2c, p, max_len=150, temp=0.7)
        print(f"Valasz: {valasz[:150]}")
