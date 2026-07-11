"""OpenSubtitles minta kiiratasa UTF-8 fajlba"""
import os

src = r'C:\NeuraNode\hemna_bench\opensubtitles_hu.txt'
dst = r'C:\NeuraNode\hemna_bench\subs_sample_utf8.txt'

# Keressuk meg a helyes kodolast
for enc in ['utf-8', 'iso-8859-2', 'latin-1', 'cp1250']:
    try:
        with open(src, 'r', encoding=enc) as f:
            lines = [f.readline() for _ in range(20)]
        # Ha van legalabb 3 latin betus sor, jo a kodolas
        ok = sum(1 for l in lines if any(c.isalpha() for c in l))
        if ok >= 3:
            print(f"Kodolas: {enc} (OK)")
            with open(dst, 'w', encoding='utf-8') as out:
                for l in lines:
                    out.write(l)
            print(f"Minta mentve: {dst}")
            
            # Mutassuk a tartalmat
            print("\n--- Minta ---")
            for l in lines[:10]:
                stipped = l.strip()
                if stipped:
                    print(f"  {stipped[:150]}")
            break
    except:
        print(f"Kodolas {enc}: nem jo")
