"""NEURA adat minta - mi van a shard-okban?"""
import torch, os
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load(r'C:\NeuraNode\bitnet\data\bitnet_pretrain\tokenizer\tokenizer.model')

DATA_DIR = r'C:\NeuraNode\bitnet\data\bitnet_pretrain\tokenized'

log_file = r'C:\NeuraNode\hemna_bench\neura_data_sample.txt'

def log(msg):
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Minden shard-bol mintavetel
for shard_idx in range(4):
    log(f"\n{'='*60}")
    log(f"SHARD {shard_idx}")
    log('='*60)
    
    d = torch.load(os.path.join(DATA_DIR, f'shard_{shard_idx}.pt'), map_location='cpu', weights_only=True)
    log(f"Shape: {d.shape}, dtype: {d.dtype}")
    log(f"Token range: [{d.min().item()}, {d.max().item()}]")
    
    # 5 random minta minden shard-bol
    torch.manual_seed(42 + shard_idx)
    indices = torch.randint(0, len(d), (5,))
    
    for i, idx in enumerate(indices):
        tokens = d[idx].tolist()
        text = sp.DecodeIds(tokens)
        log(f"\n--- Minta {i+1} (row {idx.item()}) ---")
        log(f"  Token IDk: {tokens[:15]}...")
        log(f"  Szoveg ({len(tokens)} token):")
        log(f"  {text[:400]}")
        log(f"  ---")

log(f"\n{'='*60}")
log("STATISZTIKA")
log('='*60)

# Statisztikak
all_data = torch.cat([torch.load(os.path.join(DATA_DIR, f'shard_{i}.pt'), map_location='cpu', weights_only=True) for i in range(4)], dim=0)
log(f"Osszes szekvencia: {all_data.shape[0]}")
log(f"Osszes token: {all_data.shape[0] * all_data.shape[1]:,}")

# Minta: milyen specialis tokenek vannak?
unique_tokens = set()
for i in range(5):
    row = all_data[torch.randint(0, len(all_data), (1,)).item()]
    for t in row: unique_tokens.add(t.item())
log(f"\nSpecialis tokenek (elso 500bol):")
# Keressunk specialis tokeneket
special_ids = [0, 1, 2, 3, 4, 5]  # <pad>, <unk>, <s>, </s>, stb.
for sid in special_ids:
    try:
        decoded = sp.IdToPiece(sid)
        log(f"  ID {sid}: '{decoded}'")
    except:
        pass

# Szokincsmintazott tokenek
log(f"\nSzokincs minta (elso 20 es nehany random):")
for i in range(20):
    log(f"  {i}: '{sp.IdToPiece(i)}'")
log(f"  ...")
for i in [100, 500, 1000, 5000, 10000, 20000, 30000]:
    log(f"  {i}: '{sp.IdToPiece(i)}'")

log(f"\nDONE")
