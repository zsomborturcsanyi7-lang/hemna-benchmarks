"""OpenSubtitles tokenizalasa a NEURA tokenizerrel"""
import torch, os, time
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load(r'C:\NeuraNode\bitnet\data\bitnet_pretrain\tokenizer\tokenizer.model')

SRC = r'C:\NeuraNode\hemna_bench\opensubtitles_hu.txt'
OUT_DIR = r'C:\NeuraNode\hemna_bench\subs_tokenized'
LOG_FILE = r'C:\NeuraNode\hemna_bench\tokenize_log.txt'
os.makedirs(OUT_DIR, exist_ok=True)

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    print(msg, flush=True)

SEQ_LEN = 512
SHARD_SIZE = 50000  # szekvencia per shard

log("OpenSubtitles tokenizalasa...")
log(f"Forras: {SRC}")
log(f"Meret: {os.path.getsize(SRC)/1024**3:.2f} GB")

buffer = []
shard_idx = 0
total_tokens = 0
token_count = 0
start = time.time()

def tokenize_and_save(src, out_dir, seq_len=512, shard_size=50000):
    """Sorrol sorra tokenizal, osszefuz, es ment."""
    os.makedirs(out_dir, exist_ok=True)
    
    all_tokens = []
    shard_idx = 0
    total = 0
    s = time.time()
    
    with open(src, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            ids = sp.EncodeAsIds(line)
            all_tokens.extend(ids)
            
            # Ha van eleg token, csinaljunk egy szekvenciat
            while len(all_tokens) >= seq_len:
                seq = all_tokens[:seq_len]
                all_tokens = all_tokens[seq_len:]
                
                # Folyamatosan epitjuk a shardot
                if 'shard' not in dir():
                    shard = [seq]
                else:
                    shard.append(seq)
                
                if len(shard) >= shard_size:
                    out_path = os.path.join(out_dir, f'subs_shard_{shard_idx}.pt')
                    torch.save(torch.tensor(shard, dtype=torch.int32), out_path)
                    elapsed = time.time() - s
                    log(f"Shard {shard_idx}: {len(shard)} seq ({out_path}) [{elapsed:.0f}s]")
                    shard_idx += 1
                    shard = []
                    s = time.time()
            
            if (i+1) % 100000 == 0:
                log(f"  Feldolgozva: {i+1} sor, {len(all_tokens)} pending token, {shard_idx} shard kesz")
    
    # Maradek mentese
    if 'shard' in dir() and shard:
        # Utolso szekvenciakat padding-eljuk
        while all_tokens:
            seq = all_tokens[:seq_len]
            all_tokens = all_tokens[seq_len:]
            if len(seq) < seq_len:
                seq = seq + [0] * (seq_len - len(seq))
            shard.append(seq)
        
        if shard:
            out_path = os.path.join(out_dir, f'subs_shard_{shard_idx}.pt')
            torch.save(torch.tensor(shard, dtype=torch.int32), out_path)
            log(f"Shard {shard_idx} (utolso): {len(shard)} seq")
            shard_idx += 1
    
    return shard_idx

n_shards = tokenize_and_save(SRC, OUT_DIR)
log(f"\nKESZ! {n_shards} shard, {n_shards * 50000} szekvencia")
