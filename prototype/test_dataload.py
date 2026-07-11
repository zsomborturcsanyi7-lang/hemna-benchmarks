"""Teszt: adat betoltesi ido"""
import torch, glob, time, os
t0 = time.time()
DATA_DIR = 'C:/NeuraNode/hemna_bench/combined_no_wiki'
files = sorted(glob.glob(os.path.join(DATA_DIR, '*.pt')))
print(f'Files: {len(files)}', flush=True)
t1 = time.time()
data = torch.cat([torch.load(f, map_location='cpu', weights_only=True) for f in files], dim=0)
t2 = time.time()
print(f'Data: {data.shape[0]} seq, load time: {t2-t1:.1f}s, total: {t2-t0:.1f}s', flush=True)
