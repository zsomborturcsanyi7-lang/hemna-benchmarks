"""Kombinalt adat + 300M training"""
import torch, os, glob, math
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load(r'C:\NeuraNode\bitnet\data\bitnet_pretrain\tokenizer\tokenizer.model')

# Forras mappak
SOURCES = {
    'clean': r'C:\NeuraNode\hemna_bench\clean_data',
    'subs': r'C:\NeuraNode\hemna_bench\subs_tokenized',
    'extra': r'C:\NeuraNode\hemna_bench\extra_tokenized',
}

OUT_DIR = r'C:\NeuraNode\hemna_bench\combined_data'
os.makedirs(OUT_DIR, exist_ok=True)

LOG = r'C:\NeuraNode\hemna_bench\combine_log.txt'
def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg, flush=True)

log("Adatok osszefesulese...")
total_seqs = 0
shard_idx = 0

for name, src in SOURCES.items():
    files = sorted(glob.glob(os.path.join(src, '*.pt')))
    log(f"\n{name}: {len(files)} fajl")
    
    for f in files:
        data = torch.load(f, map_location='cpu', weights_only=True)
        out_path = os.path.join(OUT_DIR, f'combined_shard_{shard_idx}.pt')
        torch.save(data, out_path)
        total_seqs += len(data)
        log(f"  Shard {shard_idx}: {len(data)} seq (ossz: {total_seqs})")
        shard_idx += 1

total_tokens = total_seqs * 512
log(f"\n{'='*50}")
log(f"OSSZESEN: {total_seqs:,} szekvencia")
log(f"OSSZESEN: {total_tokens:,} token ({total_tokens/1e9:.2f}B)")
log(f"OSSZESEN: {shard_idx} shard")
log(f"{'='*50}")
log("KESZ")
