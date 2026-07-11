"""OpenSubtitles minta kiiratasa"""
import codecs, os

src = r'C:\NeuraNode\hemna_bench\opensubtitles_hu.txt'
size = os.path.getsize(src)
print(f"Fajl meret: {size/1024**3:.2f} GB")

# Probaljuk ki a kodolast
for enc in ['utf-8', 'iso-8859-2', 'cp1250', 'latin-1']:
    try:
        f = codecs.open(src, 'r', enc)
        lines = [f.readline() for _ in range(3)]
        print(f"\nKodolas: {enc}")
        print(f"  Elso sor: {repr(lines[0][:100])}")
        f.close()
        break
    except Exception as e:
        print(f"  {enc}: {e}")
