import sys
with open(r'C:\NeuraNode\hemna_bench\train_300m_log.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    print(f"Total lines: {len(lines)}")
    for line in lines[-10:]:
        print(line.strip())
