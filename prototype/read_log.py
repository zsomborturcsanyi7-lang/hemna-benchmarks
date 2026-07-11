import sys
with open(r'C:\NeuraNode\hemna_bench\train_v2_log.txt', 'r') as f:
    lines = f.readlines()
print(f"Lines: {len(lines)}")
for l in lines:
    print(l.strip())
