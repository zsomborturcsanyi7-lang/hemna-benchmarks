"""
HEMNA v2 — Minimal Working Test
CPU-GPU Hybrid + Refinement Branch

Csak a koncepció: T1+T2 GPU-n, T3 CPU-n.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

torch.manual_seed(42)

# ============================================================
# Simplified T3 — használjunk egy egyszerű MLP-t CPU-n
# ami a nagy neuronokat szimulálja
# ============================================================
def create_hemna_v2_simple(input_dim=2, fast_dims=[16], refine_dim=4, output_dim=1):
    """Egyszerű HEMNA v2: T1+T2 GPU, T3 CPU."""
    
    class HEMNAv2Simple(nn.Module):
        def __init__(self):
            super().__init__()
            # === FAST PATH: T1 + T2 GPU-n ===
            # T1: reflex (bináris)
            self.t1 = nn.Sequential(
                nn.Linear(input_dim, 4),
                nn.Sigmoid(),
                nn.Linear(4, 4),
                nn.Sigmoid()
            ).cuda()
            # T2: normál (ReLU)
            self.t2 = nn.Sequential(
                nn.Linear(input_dim, 8),
                nn.ReLU(),
                nn.Linear(8, 4),
                nn.ReLU()
            ).cuda()
            # Konkat: 4 + 4 = 8 dim
            self.fast_head = nn.Linear(8, output_dim).cuda()
            
            # === CONFIDENCE GATE ===
            self.gate = nn.Linear(8, 1).cuda()
            
            # === SLOW PATH: T3 CPU-n ===
            self.t3 = nn.Sequential(
                nn.Linear(8, refine_dim * 2),
                nn.ReLU(),
                nn.Linear(refine_dim * 2, refine_dim),
                nn.ReLU(),
                nn.Linear(refine_dim, output_dim),
            ).cpu()
        
        def forward(self, x_gpu, threshold=0.3):
            # === FAST PATH: GPU ===
            t1_out = self.t1(x_gpu)        # [B, 4]
            t2_out = self.t2(x_gpu)        # [B, 4]
            h = torch.cat([t1_out, t2_out], dim=-1)  # [B, 8]
            
            fast_out = self.fast_head(h)   # [B, 1]
            gate_val = torch.sigmoid(self.gate(h))  # [B, 1]
            
            # === SLOW PATH: CPU ===
            # Csak ha kell
            needs = (gate_val > threshold).float()
            
            if needs.sum() > 0:
                h_cpu = h.cpu()
                refine_cpu = self.t3(h_cpu)    # CPU-n számol
                refine_gpu = refine_cpu.cuda()  # vissza GPU-ra
                
                # Csak a bizonytalan mintákra
                result = fast_out + refine_gpu * needs
            else:
                result = fast_out
                refine_gpu = None
            
            return result, gate_val, refine_gpu
    
    return HEMNAv2Simple()


# ============================================================
# XOR teszt
# ============================================================
print("=== HEMNA v2 — XOR teszt ===")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]]).float().cuda()
y = torch.tensor([[0.],[1.],[1.],[0.]]).float().cuda()

model = create_hemna_v2_simple()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

start = time.time()
for epoch in range(1000):
    pred, conf, _ = model(X, threshold=0.3)
    loss = F.mse_loss(pred, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

elapsed = time.time() - start

print(f"Tanulási idő: {elapsed:.2f}s ({elapsed/1000*1000:.2f}ms/epoch)")

print("\nEredmények:")
with torch.no_grad():
    pred, conf, refine = model(X, threshold=0.3)
    for i in range(4):
        r = "✨ T3 finomítás" if refine is not None and refine[i].abs().item() > 0.01 else ""
        print(f"  {X[i].tolist()} → {pred[i].item():.4f} "
              f"(várt: {y[i].item()})  conf={conf[i].item():.3f} {r}")

print(f"\nVégső loss: {loss.item():.6f}")

# ============================================================
# Sebesség összehasonlítás
# ============================================================
print("\n=== Sebesség benchmark ===")
batch_size = 1
X_batch = torch.randn(batch_size, 64).cuda()

# MLP reference
mlp = nn.Sequential(
    nn.Linear(64, 32), nn.ReLU(),
    nn.Linear(32, 16), nn.ReLU(),
    nn.Linear(16, 1)
).cuda()

# HEMNA v2
model2 = create_hemna_v2_simple(input_dim=64)
model2.threshold = 0.3  # néha hívja T3

# Warmup
for _ in range(20):
    mlp(X_batch)
    model2(X_batch)

torch.cuda.synchronize()
time.sleep(0.1)

# MLP
s = time.time()
for _ in range(500):
    mlp(X_batch)
torch.cuda.synchronize()
mlp_ms = (time.time() - s) / 500 * 1000

# HEMNA v2 fast (threshold=1.0 so T3 never runs)
model2.threshold = 1.0
s = time.time()
for _ in range(500):
    pred, conf, _ = model2(X_batch)
torch.cuda.synchronize()
hemna_fast_ms = (time.time() - s) / 500 * 1000

# HEMNA v2 refine (threshold=0 so T3 always runs)
model2.threshold = 0.0
s = time.time()
for _ in range(500):
    pred, conf, _ = model2(X_batch)
torch.cuda.synchronize()
hemna_refine_ms = (time.time() - s) / 500 * 1000

print(f"  MLP (GPU):          {mlp_ms:.3f} ms")
print(f"  HEMNA fast (GPU):   {hemna_fast_ms:.3f} ms")
print(f"  HEMNA refine (CPU): {hemna_refine_ms:.3f} ms")
print(f"  CPU overhead:       {hemna_refine_ms - hemna_fast_ms:.3f} ms")
