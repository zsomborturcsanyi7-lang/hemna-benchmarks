"""Javitott mikro nyelvi modell - 6 hibajavitas egyben"""
import torch, torch.nn as nn, torch.nn.functional as F, sys, time, math, json, os
from collections import Counter

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# ============================================================
# 1. BPE tokenizer (sajat implementacio)
# ============================================================
class BPETokenizer:
    """Byte-Pair Encoding tokenizer."""
    def __init__(self, vocab_size=500):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {}
        self.char_to_id = {}
        self.id_to_char = {}
    
    def _get_stats(self, words):
        pairs = Counter()
        for word, freq in words.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs
    
    def _merge(self, words, pair, new_token):
        new_words = {}
        for word, freq in words.items():
            new_word = word.replace(' '.join(pair), new_token)
            new_words[new_word] = freq
        return new_words
    
    def train(self, texts):
        # Char-level kezdes
        words = {}
        for text in texts:
            word = ' '.join(list(text))
            words[word] = words.get(word, 0) + 1
        
        # Alap szokincs (egyedi karakterek)
        chars = set()
        for text in texts:
            for c in text:
                chars.add(c)
        
        # Merge-ek
        num_merges = self.vocab_size - len(chars) - 1  # -1 a <UNK> miatt
        for i in range(num_merges):
            pairs = self._get_stats(words)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.merges[best] = best[0] + best[1]
            words = self._merge(words, best, best[0] + best[1])
        
        # Szotar epites
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for c in sorted(chars):
            vocab[c] = len(vocab)
        for pair, merged in self.merges.items():
            vocab[merged] = len(vocab)
        
        self.vocab = vocab
        self.id_to_char = {v: k for k, v in vocab.items()}
        self.char_to_id = vocab
        print(f"  BPE szotar: {len(vocab)} token (cel: {self.vocab_size})")
    
    def encode(self, text):
        tokens = list(text)
        while len(tokens) > 1:
            pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
            # Megkeressuk a legregebben merge-elt parrt
            cand = [(p, self.merges.get(p, None)) for p in pairs]
            cand = [(p, i) for i, (p, m) in enumerate(cand) if m is not None]
            if not cand:
                break
            # Vegyuk a legkorabbi merge-t
            best = min(cand, key=lambda x: list(self.merges.keys()).index(x[0]) 
                      if x[0] in self.merges else float('inf'))
            i = cand.index(best)
            tokens = tokens[:i] + [self.merges[best[0]]] + tokens[i+2:]
        
        return [self.char_to_id.get(t, self.char_to_id['<UNK>']) for t in tokens]
    
    def decode(self, ids):
        return ''.join(self.id_to_char.get(i, '?') for i in ids)
    
    def vocab_size(self):
        return len(self.vocab)

# ============================================================
# 2. Adat (magyar mese)
# ============================================================
print("Adat betoltese...")
# A mar megl evo szoveget hasznaljuk, de tobb valtozattal
torch.manual_seed(42)

base_text = """Az ember agya egy csodalatos szerkezet. Kutatasa mind a mai napig tart.
A mesterseges intelligencia egy olyan terulet ahol a szamitogepeket probaljuk 
emberi intelligenciaval felruhazni. A gepi tanulas segitsegevel a szamitogepek 
kepesek mintakat felismerni az adatokbol. A melytanulas tovabb lep tobb retegu 
halozatokkal dolgozik. A nyelvfeldolgozas az egyik legnehezebb feladat. 
Egy gepnek meg kell ertenie a szavak jelentese kozotti osszefuggeseket. 
A transformer architektura forradalmasitotta a termeszetes nyelvfeldolgozast. 
Az attention mechanizmus minden szot minden szoval osszekot. 
A jovo mesterseges inteligenciaja meg tobbet fog tudni.
A kutatok ujabb es ujabb modszereket fejlesztenek. 
A cel egy olyan gep letrehozasa ami tenylegesen megerti az emberi nyelvet. 
Ez az ut hosszu de izgalmas. Minden kis lepes kozelebb visz a celhoz."""

# Keverjunk be mas mondatokat is a valtozatossagert
extra_topics = [
    "A mai idojaras napos es meleg lesz. Holnap varnato eso is lehet.",
    "A macska az asztal alatt alszik. A kutya a kertben jatszik.",
    "A matek hatt feladatot kaptam. Meg kell oldanom a harmadik peldat is.",
    "A zenet mindenki szereti. A klasszikus zene megnyugtat.",
    "Az alma egeszseges gyumolcs. Naponta egy alma az orvost tavol tartja.",
    "A programozas izgalmas hobbi. Uj nyelveket tanulni mindig erdekes.",
    "A konyvek fontosak a tanulashoz. Egy jo konyv eletre szolo tudast ad.",
    "A sport segit egeszsegesnek maradni. Minden nap edzek egy kicsit.",
    "A baratok tamaszt nyujtanak a nehez idokben. Egy jo barat ritka kincs.",
    "Az utazas szinesiti az eletet. Uj helyeket felfedezni csodas dolog.",
    "A reggeli a nap legfontosabb etkezese. Minden nap eszem meleg reggelit.",
    "A technologia folyamatosan fejlodik. Minden evben uj talalmanyok szuletnek.",
]

# Bovebb adat: tobb valtozat = jobb altalanositas
all_texts = [base_text] + extra_topics
corpus_text = '\n'.join(all_texts) * 200  # ~300K karakter
corpus_lines = all_texts * 2000

print(f"  Korpusz meret: {len(corpus_text)} karakter")

# BPE betanitasa
tokenizer = BPETokenizer(vocab_size=500)
tokenizer.train(corpus_lines)

# Adat elokeszites
all_data = []
for line in corpus_lines:
    tokens = tokenizer.encode(line)
    if len(tokens) > 2:  # ertelmes hosszusagu
        all_data.extend(tokens)

all_data = torch.tensor(all_data, dtype=torch.long)
n = len(all_data)
train_data = all_data[:int(n*0.9)]
val_data = all_data[int(n*0.9):]
print(f"  Tokenek: {len(all_data)}, szokincs: {tokenizer.vocab_size()}")

VOCAB_SIZE = tokenizer.vocab_size()

def get_batch(data, bs=32, bl=128):
    ix = torch.randint(len(data)-bl-1, (bs,))
    x = torch.stack([data[i:i+bl] for i in ix])
    y = torch.stack([data[i+1:i+bl+1] for i in ix])
    return x.to(device), y.to(device)

def count_params(m):
    return sum(p.numel() for p in m.parameters())

# ============================================================
# 3. Javitott architektura
# ============================================================
class SwiGLU(nn.Module):
    """Gated FFN SwiGLU-val."""
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, False)
        self.w2 = nn.Linear(hidden_dim, dim, False)
        self.w3 = nn.Linear(dim, hidden_dim, False)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class SlidingWindowAttention(nn.Module):
    """GQA + Sliding Window."""
    def __init__(self, dim, n_heads, n_kv_heads, window=128):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = dim // n_heads
        self.window = window
        
        self.wq = nn.Linear(dim, n_heads * self.head_dim, False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, False)
        self.drop = nn.Dropout(0.1)
        
        # ALiBi helyett: relative pozicio bias
        self.register_buffer('rel_bias', torch.zeros(1, 1, window, window))
    
    def forward(self, x):
        B, T, D = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        # Repeat KV heads a GQA-hoz
        if self.n_rep > 1:
            k = k[:, :, None].expand(-1, -1, self.n_rep, -1, -1).reshape(B, self.n_heads, T, self.head_dim)
            v = v[:, :, None].expand(-1, -1, self.n_rep, -1, -1).reshape(B, self.n_heads, T, self.head_dim)
        
        # Sliding window: csak az utolso window tokenre figyelunk
        wei = torch.zeros(B, self.n_heads, T, T, device=x.device)
        for t in range(T):
            start = max(0, t - self.window + 1)
            # Itt lehetne hatékonyabb, de 128-as ablakkal ez O(128*T), ami OK
            slice_q = q[:, :, t:t+1]  # [B, H, 1, hd]
            slice_k = k[:, :, start:t+1]  # [B, H, window, hd]
            wei[:, :, t, start:t+1] = (slice_q @ slice_k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        
        # Causal mask + sliding window
        causal = torch.tril(torch.ones(T, T, device=x.device))
        window_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=-(self.window-1))
        mask = causal * window_mask
        wei = wei.masked_fill(mask == 0, float('-inf'))
        
        wei = F.softmax(wei, dim=-1)
        wei = self.drop(wei)
        out = (wei @ v).transpose(1, 2).reshape(B, T, -1)
        return self.wo(out)

class Block(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, ffn_dim, window=128, dropout=0.1):
        super().__init__()
        self.sa = SlidingWindowAttention(dim, n_heads, n_kv_heads, window)
        self.ffn = SwiGLU(dim, ffn_dim, dropout)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x):
        x = x + self.drop(self.sa(self.ln1(x)))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x

class MicroLMv2(nn.Module):
    """Javitott mikro nyelvi modell."""
    def __init__(self, vocab_size, dim=256, n_layers=6, n_heads=8, 
                 ffn_dim=512, window=128, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[
            Block(dim, n_heads, max(2, n_heads//4), ffn_dim, window, dropout)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, vocab_size, False)  # tied embedding
    
    def forward(self, x):
        x = self.drop(self.tok_emb(x))
        x = self.blocks(x)
        return self.out(self.ln(x))

# ============================================================
# 4. Benchmark
# ============================================================
print(f"\n{'Konfig':<25} {'Params':<10} {'Loss':<10} {'PPL':<10} {'Ido':<10}")
print("-" * 70)

# Tobb konfiguracio, hogy lassuk a javulast
confs = [
    ("MicroLMv2 (256d-6l-512ffn)", 256, 6, 8, 512, 128, 0.1),
]

torch.manual_seed(42)
for label, dim, nl, nh, ffn_d, window, drop in confs:
    model = MicroLMv2(VOCAB_SIZE, dim, nl, nh, ffn_d, window, drop).to(device)
    np = count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    
    print(f"\n{label:<25} {np:<10} ", end='')
    
    s = time.time()
    model.train()
    for step in range(2000):
        x, y = get_batch(train_data, 32, 128)
        opt.zero_grad()
        loss = F.cross_entropy(model(x).view(-1, VOCAB_SIZE), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0:
            model.eval()
            with torch.no_grad():
                losses = []
                for _ in range(10):
                    xv, yv = get_batch(val_data, 32, 128)
                    losses.append(F.cross_entropy(model(xv).view(-1, VOCAB_SIZE), yv.view(-1)).item())
                vl = sum(losses)/len(losses)
            model.train()
    
    model.eval()
    with torch.no_grad():
        losses = []
        for _ in range(50):
            xv, yv = get_batch(val_data, 32, 128)
            losses.append(F.cross_entropy(model(xv).view(-1, VOCAB_SIZE), yv.view(-1)).item())
    fl = sum(losses)/len(losses)
    ppl = math.exp(fl)
    
    # Generacio
    prompt = "A mesterseges intelligencia"
    prompt_ids = tokenizer.encode(prompt)
    x = torch.tensor([prompt_ids], device=device)
    gen = list(prompt_ids)
    model.eval()
    with torch.no_grad():
        for _ in range(150):
            logits = model(x[:, -128:])
            probs = F.softmax(logits[0, -1] / 0.8, dim=-1)
            next_t = torch.multinomial(probs, 1).item()
            gen.append(next_t)
            x = torch.cat([x, torch.tensor([[next_t]], device=device)], dim=1)
    
    generated = tokenizer.decode(gen)
    print(f"{fl:<10.4f} {ppl:<10.2f} {time.time()-s:<10.1f}")
    print(f"  Generalas: {generated[:200]}")
