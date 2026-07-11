"""NEURA adat tisztitas - magyar nyelvu adat szurese"""
import torch, os, re, time, math
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load(r'C:\NeuraNode\bitnet\data\bitnet_pretrain\tokenizer\tokenizer.model')

DATA_DIR = r'C:\NeuraNode\bitnet\data\bitnet_pretrain\tokenized'
OUT_DIR = r'C:\NeuraNode\hemna_bench\clean_data'
os.makedirs(OUT_DIR, exist_ok=True)

LOG_FILE = r'C:\NeuraNode\hemna_bench\clean_data_log.txt'
def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    print(msg)

# Angol szavak listaja (gyakori angol tokenek)
ENGLISH_WORDS = {'the', 'is', 'are', 'was', 'were', 'this', 'that', 'with', 'from', 'have', 
                 'been', 'also', 'its', 'has', 'had', 'but', 'not', 'for', 'and', 'can',
                 'will', 'would', 'could', 'should', 'may', 'than', 'which', 'their',
                 'there', 'these', 'those', 'after', 'before', 'between', 'through',
                 'during', 'without', 'because', 'however', 'therefore', 'about',
                 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
                 'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both',
                 'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own',
                 'same', 'so', 'too', 'very', 'just', 'also', 'now', 'even',
                 'still', 'already', 'always', 'never', 'often', 'usually',
                 'well', 'here', 'there', 'almost', 'enough', 'quite', 'rather',
                 'really', 'nearly', 'fully', 'deeply', 'strongly', 'highly',
                 'article', 'page', 'edit', 'source', 'retrieved', 'archived',
                 'original', 'accessed', 'published', 'updated', 'website',
                 'newspaper', 'magazine', 'journal', 'volume', 'issue', 'doi',
                 'isbn', 'issn', 'pmid', 'pmc', 'oclc', 'wikipedia', 'commons',
                 'wikimedia', 'wikidata', 'category', 'template', 'portal'}

# Magyar specialis karakterek
HUNGARIAN_CHARS = set('áéíóöőúüűÁÉÍÓÖŐÚÜŰ')

def is_hungarian(text):
    """Magyar nyelv detektalasa karakterek alapjan."""
    if len(text) < 10:
        return False
    # Magyar specialis karakterek szamlalasa
    hu_count = sum(1 for c in text if c in HUNGARIAN_CHARS)
    # Angol szavak szamlalasa
    words = text.lower().split()
    en_count = sum(1 for w in words if w.rstrip('.,;:!?()[]') in ENGLISH_WORDS)
    
    # Ha >1% magyar karakter es <10% angol szav -> magyar
    hu_ratio = hu_count / len(text)
    en_ratio = en_count / max(len(words), 1)
    
    return hu_ratio > 0.008 and en_ratio < 0.12

def clean_text(text):
    """Szoveg tisztitasa."""
    # Labjegyzetek kiszedese [1], [2], [a], [b], stb.
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[[\d\s]+\]', '', text)
    # Nyilak es specialis karakterek
    text = text.replace('↑', '').replace('→', '').replace('↓', '').replace('←', '')
    # Tobb space osszevonasa
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_structured_data(text):
    """Tablazatos/tul strukturalt adat detektalasa."""
    if len(text) < 20:
        return True
    # Sok szam (datumok, sporteredmenyek, statisztikak)
    digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
    if digit_ratio > 0.20:
        return True
    # Sok specialis karakter (tablazatok)
    special_ratio = sum(1 for c in text if c in '|=-+*/_\\') / max(len(text), 1)
    if special_ratio > 0.05:
        return True
    return False

def has_too_much_english(ids):
    """Token szintu angol szures."""
    total = len(ids)
    if total == 0:
        return True
    eng_count = 0
    for tid in ids[:100]:  # elso 100 token eleg
        try:
            piece = sp.IdToPiece(tid).lower().strip('▁')
            if piece in ENGLISH_WORDS:
                eng_count += 1
        except:
            pass
    # Ha >15% angol -> eldob
    return eng_count / min(total, 100) > 0.15

# ============================================================
# Futtatas
# ============================================================
log("NEURA adat tisztitas elkezdve...")
log(f"Angol szavak listaja: {len(ENGLISH_WORDS)} szo")
log(f"Magyar karakterek: {''.join(HUNGARIAN_CHARS)}")

total_before = 0
total_after = 0
stats = {'total': 0, 'hungarian': 0, 'english_removed': 0, 'structured_removed': 0, 'cleaned': 0}

for shard_idx in range(4):
    log(f"\n--- Shard {shard_idx} betoltese ---")
    d = torch.load(os.path.join(DATA_DIR, f'shard_{shard_idx}.pt'), map_location='cpu', weights_only=True)
    n = len(d)
    total_before += n
    stats['total'] += n
    
    cleaned = []
    for i in range(n):
        tokens = d[i].tolist()
        text = sp.DecodeIds(tokens)
        
        # 1. Magyar nyelv ellenorzese
        if not is_hungarian(text[:300]):  # elso 300 karakter eleg
            stats['english_removed'] += 1
            continue
        
        # 2. Strukturalt adat szures
        cleaned_text = clean_text(text)
        if is_structured_data(cleaned_text):
            stats['structured_removed'] += 1
            continue
        
        # 3. Token szintu angol szures
        if has_too_much_english(tokens):
            stats['english_removed'] += 1
            continue
        
        cleaned.append(tokens)
        stats['hungarian'] += 1
        
        if (i+1) % 10000 == 0:
            log(f"  Feldolgozva: {i+1}/{n}, eddig magyar: {stats['hungarian']}")
    
    # Ujratokenizaljuk a tisztitott szoveget es mentjuk
    log(f"  Shard {shard_idx}: {n} -> {len(cleaned)} szekvencia")
    
    if cleaned:
        # Re-tokenize the cleaned texts
        cleaned_data = []
        for tokens in cleaned:
            text = sp.DecodeIds(tokens)
            text = clean_text(text)
            new_ids = sp.EncodeAsIds(text)
            if len(new_ids) >= 50:  # minimum hossz
                # Tobbitsunk vagy vagjuk 512-re
                if len(new_ids) < 512:
                    new_ids = new_ids + [0] * (512 - len(new_ids))
                else:
                    new_ids = new_ids[:512]
                cleaned_data.append(new_ids)
        
        out_tensor = torch.tensor(cleaned_data, dtype=torch.int32)
        out_path = os.path.join(OUT_DIR, f'shard_{shard_idx}_clean.pt')
        torch.save(out_tensor, out_path)
        log(f"  Mentve: {out_path} ({len(out_tensor)} szekvencia)")
        stats['cleaned'] += len(out_tensor)

log(f"\n{'='*60}")
log(f"STATISZTIKA")
log(f"{'='*60}")
log(f"Osszes szekvencia: {stats['total']}")
log(f"  - Magyar: {stats['hungarian']} ({stats['hungarian']/stats['total']*100:.1f}%)")
log(f"  - Angol/egyeb eltavolitva: {stats['english_removed']} ({stats['english_removed']/stats['total']*100:.1f}%)")
log(f"  - Strukturalt adat eltavolitva: {stats['structured_removed']} ({stats['structured_removed']/stats['total']*100:.1f}%)")
log(f"  - Vegleges tiszta szekvencia: {stats['cleaned']}")
log(f"DONE")
