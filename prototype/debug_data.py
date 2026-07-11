import torch
import glob, os

# Kereses
for root, dirs, files in os.walk(r'C:\NeuraNode\bitnet\data'):
    for f in files:
        if f.endswith('.pt'):
            print(f"File: {os.path.join(root, f)}")
            size = os.path.getsize(os.path.join(root, f))
            print(f"  Size: {size:,} bytes")
            if size < 700_000_000:
                try:
                    d = torch.load(os.path.join(root, f), map_location='cpu', weights_only=True)
                    print(f"  Type: {type(d)}")
                    if isinstance(d, dict):
                        print(f"  Keys: {list(d.keys())[:5]}")
                    elif isinstance(d, torch.Tensor):
                        print(f"  Shape: {d.shape}, dtype={d.dtype}")
                        print(f"  First 20: {d[:20].tolist()}")
                except Exception as e:
                    print(f"  Error: {e}")
