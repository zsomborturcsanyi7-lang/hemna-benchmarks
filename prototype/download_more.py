"""HunSum-2 + MEK konyvek letoltese"""
import torch, os, time, urllib.request
from datasets import load_dataset
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load(r'C:\NeuraNode\bitnet\data\bitnet_pretrain\tokenizer\tokenizer.model')

OUT = r'C:\NeuraNode\hemna_bench\extra_tokenized'
LOG_FILE = r'C:\NeuraNode\hemna_bench\download_log.txt'
os.makedirs(OUT, exist_ok=True)

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    print(msg, flush=True)

SEQ_LEN = 512
SHARD_SIZE = 50000

def tokenize_and_save(text_stream, out_dir, name, max_items=None):
    buffer, shard, shard_idx = [], [], 0
    s = time.time()
    for i, text in enumerate(text_stream):
        if max_items and i >= max_items: break
        text = text.strip()
        if len(text) < 20: continue
        ids = sp.EncodeAsIds(text)
        if len(ids) < 5: continue
        buffer.extend(ids)
        while len(buffer) >= SEQ_LEN:
            shard.append(buffer[:SEQ_LEN])
            buffer = buffer[SEQ_LEN:]
            if len(shard) >= SHARD_SIZE:
                torch.save(torch.tensor(shard, dtype=torch.int32),
                           os.path.join(out_dir, f'{name}_shard_{shard_idx}.pt'))
                log(f"  {name} shard {shard_idx}: {len(shard)} seq [{time.time()-s:.0f}s]")
                shard_idx += 1; s = time.time(); shard = []
        if (i+1) % 50000 == 0:
            log(f"  {name}: {i+1} items, {shard_idx} shards")
    if shard:
        while buffer:
            seq = buffer[:SEQ_LEN]; buffer = buffer[SEQ_LEN:]
            if len(seq) < SEQ_LEN: seq = seq + [0]*(SEQ_LEN-len(seq))
            shard.append(seq)
        torch.save(torch.tensor(shard, dtype=torch.int32),
                   os.path.join(out_dir, f'{name}_shard_{shard_idx}.pt'))
        log(f"  {name} shard {shard_idx} (last): {len(shard)} seq"); shard_idx += 1
    return shard_idx

# ============================================================
# 1. HunSum-2
# ============================================================
log("\n=== HUNSUM-2 ===")
ds = load_dataset('SZTAKI-HLT/HunSum-2-abstractive', split='train', streaming=True)
log("Streaming HunSum-2...")
n = tokenize_and_save((item['article'] for item in ds), OUT, 'hunsum2')
log(f"HunSum-2 DONE: {n} shards")

# ============================================================
# 2. MEK konyvek
# ============================================================
log("\n=== MEK KONYVEK ===")
MEK_BOOKS = [
    ('00000/00056', 'Jokai: A koszivu ember fiai'),
    ('00000/00058', 'Arany: Toldi'),
    ('00000/00104', 'Madach: Az ember tragediaja'),
    ('00000/00016', 'Petofi: Osszes koltemenyei'),
    ('00000/00057', 'Jokai: Az arany ember'),
    ('00000/00059', 'Arany: Toldi esteje'),
    ('00000/00060', 'Arany: Buda halala'),
    ('00000/00018', 'Mikszath: A tot atyafiak'),
    ('00200/00286', 'Karinthy: Tanar ur kerem'),
    ('00000/00142', 'Moricz: Legy jo mindhalalig'),
    ('00000/00335', 'Gardonyi: Egri csillagok'),
    ('00000/00102', 'Mikszath: Szent Peter esernyoje'),
    ('00000/00020', 'Babits: A golyakalifa'),
    ('00000/00160', 'Krudy: Szindbad'),
    ('00000/00336', 'Kosztolanyi: Edes Anna'),
]

all_texts = []
for code, title in MEK_BOOKS:
    code_num = code.split('/')[1]
    url = f'https://mek.oszk.hu/{code}/{code_num}.txt'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as f:
            text = f.read().decode('utf-8', errors='replace')
        all_texts.append(text)
        log(f"  OK: {title} ({len(text)} bytes)")
    except Exception as e:
        log(f"  FAIL: {title}: {e}")

if all_texts:
    combined = '\n\n'.join(all_texts)
    log(f"MEK total: {len(combined)} bytes")
    n = tokenize_and_save([combined], OUT, 'mek')
    log(f"MEK DONE: {n} shards")

log("\n=== ALL DONE ===")
