"""Kibontja az OpenSubtitles magyar adatot"""
import gzip, os

src = r'C:\Users\neura\opensubtitles_hu.txt.gz'
dst = r'C:\NeuraNode\hemna_bench\opensubtitles_hu.txt'

print(f"Forras: {src}")
print(f"Meret: {os.path.getsize(src)/1024**3:.2f} GB")

print("Kibontas...")
buf = gzip.open(src, 'rb').read()
open(dst, 'wb').write(buf)

print(f"Kibontva: {len(buf)} bytes = {len(buf)/1024**3:.2f} GB")

# Szamold a sorokat
lines = buf.decode('utf-8', errors='ignore').split('\n')
print(f"Sorok: {len(lines):,}")

# Mintavetel
print("\n--- Elso 5 sor ---")
for l in lines[:5]:
    print(f"  {l[:100]}")
print("\n--- Utolso 5 sor ---")
for l in lines[-6:-1]:
    print(f"  {l[:100]}")
