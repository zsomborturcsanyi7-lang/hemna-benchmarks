"""Combined adat Wiki nelkul + training folytatas"""
import torch, os, glob, math, time
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load(r'C:\NeuraNode\bitnet\data\bitnet_pretrain\tokenizer\tokenizer.model')
V = sp.GetPieceSize()

OUT_DIR = r'C:\NeuraNode\hemna_bench\combined_no_wiki'
os.makedirs(OUT_DIR, exist_ok=True)

LOG = r'C:\NeuraNode\hemna_bench\combine_no_wiki_log.txt'
def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    print(msg, flush=True)

# CSAK subs + hunsum2 + mek (nincs NEURA/wiki)
SOURCES = {
    'subs': r'C:\NeuraNode\hemna_bench\subs_tokenized',  # 55 shard
    'extra': r'C:\NeuraNode\hemna_bench\extra_tokenized',  # 46 shard (hunsum2 + mek)
}

log("Combined adat Wiki nelkul...")
total_seqs = 0
shard_idx = 0

for name, src in SOURCES.items():
    files = sorted(glob.glob(os.path.join(src, '*.pt')))
    log(f"{name}: {len(files)} fajl")
    
    for f in files:
        data = torch.load(f, map_location='cpu', weights_only=True)
        out_path = os.path.join(OUT_DIR, f'combined_{shard_idx}.pt')
        torch.save(data, out_path)
        total_seqs += len(data)
        if shard_idx % 10 == 0:
            log(f"  Shard {shard_idx}: {total_seqs:,} seq")
        shard_idx += 1

total_tokens = total_seqs * 512
log(f"\nOSSZES: {total_seqs:,} seq, {total_tokens:,} token ({total_tokens/1e9:.2f}B)")
log(f"Shardok: {shard_idx}")
log("KESZ")
