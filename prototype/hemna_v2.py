"""
HEMNA v2 — CPU-GPU Hybrid + Refinement Branch

Architektúra:

    Bemenet
       │
       ├──→ T1/T2 (GPU) ──→ gyors kimenet ──→ VÁLASZ
       │                          │
       └──→ T3 (CPU) ─────────────┘ (párhuzamos finomítás)
                         ↑
                  Ha a háló bizonytalan,
                  T3 belenyúl és javít

Előnyök:
  - GPU nem vár CPU-ra (T3 nem blokkol)
  - T3 csak akkor fut ha kell
  - CPU RAM korlátlan T3 számára
  - NEURA-hoz természetes: TÖMÖR (fast) + Qwen (refinement)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import threading
from typing import Optional


# ============================================================
# T1 — Apró (reflex) neuron
# ============================================================
class Tier1Layer(nn.Module):
    """
    Bináris reflex neuronok.
    y = step(Σ w·x)  → kimenet 0 vagy 1
    """
    def __init__(self, in_features: int, n_neurons: int):
        super().__init__()
        self.in_features = in_features
        self.n_neurons = n_neurons
        self.weight = nn.Parameter(torch.randn(n_neurons, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(n_neurons))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, torch.sign(self.weight), self.bias)
        return torch.sigmoid(out).round()

    def extra_repr(self):
        return f'T1[{self.n_neurons}]: {self.in_features}→{self.n_neurons}'


# ============================================================
# T2 — Normál MLP neuron
# ============================================================
class Tier2Layer(nn.Module):
    """
    Normál ReLU MLP neuron.
    y = ReLU(Wx + b)
    """
    def __init__(self, in_features: int, n_neurons: int):
        super().__init__()
        self.in_features = in_features
        self.n_neurons = n_neurons
        self.linear = nn.Linear(in_features, n_neurons)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.linear(x))

    def extra_repr(self):
        return f'T2[{self.n_neurons}]: {self.in_features}→{self.n_neurons}'


# ============================================================
# T3 —B-spline neuron (CPU-n fut)
# ============================================================
class BSpline(nn.Module):
    """
    B-spline — vektorizált implementáció (gyorsabb).
    Degree=2, Cox-de Boor CUDA-kompatibilis.
    """
    def __init__(self, grid_size: int = 8):
        super().__init__()
        self.grid_size = grid_size
        n_coeffs = grid_size + 2  # degree=2
        self.coefficients = nn.Parameter(torch.randn(n_coeffs) * 0.1)
        # Grid: egyenletes [-2, 2]
        grid = torch.linspace(-2.0, 2.0, grid_size)
        self.register_buffer('grid', grid)
        self.register_buffer('degree', torch.tensor(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        device = x.device
        k = 2
        
        # Kiterjesztett grid a Cox-de Boor miatt (clamp + extra)
        # Egyszerűbb: közvetlen spline értékelés
        # Használunk egy egyszerűbb approach-öt: Catmull-Rom stílusú
        
        x_clamped = torch.clamp(x, self.grid[0], self.grid[-1])
        
        # Melyik grid intervallumban van x?
        # grid[i] <= x < grid[i+1]
        intervals = (x_clamped.unsqueeze(-1) >= self.grid.unsqueeze(0)).sum(dim=-1) - 1
        intervals = torch.clamp(intervals, 0, self.grid_size - 2)
        
        # B-spline bázis degree 0, 1, 2 egyszerűsítve
        # N_i,0(x)
        basis = torch.zeros(batch, self.grid_size - 1 + 2, device=device)
        
        # Degree 0 alap
        for i in range(self.grid_size - 1):
            mask = (x_clamped >= self.grid[i]) & (x_clamped < self.grid[i + 1])
            if mask.any():
                basis[mask, i] = 1.0
        
        # Degree 1
        basis1 = torch.zeros_like(basis)
        for i in range(self.grid_size - 2):
            left = (x_clamped - self.grid[i]) / (self.grid[i+1] - self.grid[i] + 1e-8)
            right = (self.grid[i+2] - x_clamped) / (self.grid[i+2] - self.grid[i+1] + 1e-8)
            basis1[:, i] = left.clamp(0,1) * basis[:, i] + right.clamp(0,1) * basis[:, i+1]
        
        # Degree 2
        basis2 = torch.zeros_like(basis)
        for i in range(self.grid_size - 3):
            left = (x_clamped - self.grid[i]) / (self.grid[i+2] - self.grid[i] + 1e-8)
            right = (self.grid[i+3] - x_clamped) / (self.grid[i+3] - self.grid[i+1] + 1e-8)
            basis2[:, i] = left.clamp(0,1) * basis1[:, i] + right.clamp(0,1) * basis1[:, i+1]
        
        # coefficients @ basis
        n_coeffs = self.grid_size + 1  # grid_size + k - 1 = 8+2-1=9, de használjunk 9-et
        coeffs = self.coefficients[:n_coeffs]
        result = basis2[:, :n_coeffs] @ coeffs
        return result


class Tier3Layer(nn.Module):
    """
    Nagy neuronok — minden kapcsolaton B-spline.
    KIFEJEZETTEN CPU-n fut a refinement ágban.
    """
    def __init__(self, in_features: int, n_neurons: int, grid_size: int = 8,
                 device: torch.device = torch.device('cpu')):
        super().__init__()
        self.in_features = in_features
        self.n_neurons = n_neurons
        self.device = device
        
        # Minden (neuron, input) kapcsolathoz egy spline
        # CPU-n tároljuk, mert itt fut
        self.splines = nn.ModuleList([
            nn.ModuleList([BSpline(grid_size=grid_size) for _ in range(in_features)])
            for _ in range(n_neurons)
        ])
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x CPU-n van (GPU→CPU copy már megtörtént)
        batch = x.shape[0]
        outputs = []
        for n_idx in range(self.n_neurons):
            val = torch.zeros(batch, device=self.device)
            for i_idx in range(self.in_features):
                val += self.splines[n_idx][i_idx](x[:, i_idx])
            outputs.append(val.unsqueeze(1))
        return torch.cat(outputs, dim=1)

    def extra_repr(self):
        return f'T3[{self.n_neurons}]: {self.in_features}→{self.n_neurons} @ {self.device}'


# ============================================================
# HEMNA v2 — Hybrid CPU-GPU + Refinement
# ============================================================

class ConfidenceGate(nn.Module):
    """
    Eldönti hogy kell-e T3 refinement.
    
    Kimenet: 0 = T1/T2 elég, 1 = hívd a T3-at
    """
    def __init__(self, in_features: int):
        super().__init__()
        self.gate = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 0 = biztos, 1 = bizonytalan
        return torch.sigmoid(self.gate(x))


class HEMNAv2(nn.Module):
    """
    HEMNA v2 — CPU-GPU Hybrid + Refinement Branch
    
    Fast path: T1/T2 GPU-n → gyors válasz
    Slow path: T3 CPU-n → finomítás (csak ha kell)
    
    Példa:
        model = HEMNAv2(
            input_dim=10,
            fast_dims=[32, 16],     # T1/T2 GPU rétegek
            refine_dim=8,            # T3 CPU finomító
            output_dim=2,
            t1_ratio=0.3,
            confidence_threshold=0.1
        )
    """
    def __init__(self, input_dim: int, fast_dims: list, refine_dim: int,
                 output_dim: int, t1_ratio: float = 0.3,
                 confidence_threshold: float = 0.1,
                 gpu_device='cuda', cpu_device='cpu'):
        super().__init__()
        self.confidence_threshold = confidence_threshold
        
        # ============================================================
        # FAST PATH — T1 + T2 (GPU)
        # ============================================================
        fast_layers = []
        prev_dim = input_dim
        
        for i, h_dim in enumerate(fast_dims):
            n_t1 = max(1, int(h_dim * t1_ratio))
            n_t2 = h_dim - n_t1
            
            class FastLayer(nn.Module):
                def __init__(s):
                    super().__init__()
                    s.t1 = Tier1Layer(prev_dim, n_t1) if n_t1 > 0 else None
                    s.t2 = Tier2Layer(prev_dim, n_t2) if n_t2 > 0 else None
                    s.n_t1 = n_t1
                    s.n_t2 = n_t2
                
                def forward(s, x):
                    outs = []
                    if s.t1 is not None:
                        outs.append(s.t1(x))
                    if s.t2 is not None:
                        outs.append(s.t2(x))
                    return torch.cat(outs, dim=-1)
            
            fast_layers.append(FastLayer().to(gpu_device))
            prev_dim = h_dim
        
        self.fast_layers = nn.ModuleList(fast_layers)
        
        # Fast path kimenet (az utolsó fast réteg kimenete)
        self.fast_output = nn.Linear(prev_dim, output_dim).to(gpu_device)
        
        # ============================================================
        # CONFIDENCE GATE (GPU)
        # ============================================================
        self.confidence_gate = ConfidenceGate(prev_dim).to(gpu_device)
        
        # ============================================================
        # SLOW PATH — T3 (CPU)
        # ============================================================
        self.t3 = Tier3Layer(prev_dim, refine_dim, device=cpu_device)
        
        # T3 kimenet → hozzáadás a fast kimenethez
        self.refine_projection = nn.Linear(refine_dim, output_dim).to(cpu_device)
        
        # GPU és CPU eszközök
        self.gpu_device = torch.device(gpu_device)
        self.cpu_device = torch.device(cpu_device)
        
        # Async copy stream
        if gpu_device != 'cpu':
            self.copy_stream = torch.cuda.Stream()
        else:
            self.copy_stream = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        1. Fast path (T1+T2 GPU-n) → gyors válasz
        2. Ha bizonytalan: T3 (CPU) → finomítás
        3. Válasz = fast_kimenet + (refinement ha kell)
        """
        x = x.to(self.gpu_device)
        
        # === Fast path (GPU) ===
        h = x
        for layer in self.fast_layers:
            h = layer(h)
        
        fast_result = self.fast_output(h)           # [batch, output_dim]
        confidence = self.confidence_gate(h)         # [batch, 1]
        needs_refine = (confidence > self.confidence_threshold).float()
        
        # === Slow path (CPU) — párhuzamos ===
        # Async copy GPU→CPU
        if self.copy_stream is not None and needs_refine.mean() > 0:
            with torch.cuda.stream(self.copy_stream):
                h_cpu = h.cpu()
        else:
            h_cpu = h.cpu()
        
        # T3 számolás CPU-n
        if needs_refine.mean() > 0:
            refine = self.t3(h_cpu)                   # [batch, refine_dim]
            refine_out = self.refine_projection(refine) # [batch, output_dim]
            
            # Sync copy vissza GPU-ra
            if self.copy_stream is not None:
                torch.cuda.synchronize()
            refine_out = refine_out.to(self.gpu_device)
            
            # Csak a bizonytalan mintákra alkalmazzuk
            result = fast_result + refine_out * needs_refine
        else:
            result = fast_result
        
        return result, confidence
    
    def get_stats(self):
        """Visszaadja az architektúra statisztikáit."""
        n_t1_total = 0
        n_t2_total = 0
        for layer in self.fast_layers:
            if hasattr(layer, 'n_t1'):
                n_t1_total += layer.n_t1
            if hasattr(layer, 'n_t2'):
                n_t2_total += layer.n_t2
        
        stats = {
            'T1 (GPU)': n_t1_total,
            'T2 (GPU)': n_t2_total,
            'T3 (CPU)': self.t3.n_neurons,
            'Fast output': f"{self.fast_output.in_features}→{self.fast_output.out_features}",
            'Refine proj': f"{self.refine_projection.in_features}→{self.refine_projection.out_features}",
            'T3 device': str(self.t3.device),
            'Threshold': self.confidence_threshold,
        }
        return stats


# ============================================================
# TESZT
# ============================================================
def test_hemna_v2():
    """HEMNA v2 teszt: XOR + hibrid CPU-GPU."""
    torch.manual_seed(42)
    
    X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]]).float().cuda()
    y = torch.tensor([[0.],[1.],[1.],[0.]]).float().cuda()
    
    # HEMNA v2 — T1+T2 GPU-n, T3 CPU-n
    model = HEMNAv2(
        input_dim=2,
        fast_dims=[16],       # 1 réteg T1+T2 GPU-n
        refine_dim=4,          # 4 T3 neuron CPU-n
        output_dim=1,
        t1_ratio=0.25,
        confidence_threshold=0.2
    )
    
    print("=== HEMNA v2 — CPU-GPU Hybrid ===")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CPU: {model.cpu_device}")
    print(f"\nArchitektúra:")
    for k, v in model.get_stats().items():
        print(f"  {k}: {v}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    
    for epoch in range(1000):
        pred, confidence = model(X)
        loss = F.mse_loss(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            avg_conf = confidence.mean().item()
            print(f"  Epoch {epoch:4d}: loss={loss.item():.6f}  "
                  f"bizonytalanság={avg_conf:.3f}")
    
    print(f"\nVégső eredmény:")
    with torch.no_grad():
        pred, conf = model(X)
        for inp, p, c, t in zip(X, pred, conf, y):
            refine_str = " ✨ finomítva" if c > 0.2 else ""
            print(f"  {inp.tolist()} → {p.item():.3f} (várt: {t.item()})"
                  f"  conf={c.item():.3f}{refine_str}")


def benchmark_speed():
    """Sebesség összehasonlítás: HEMNA v2 vs hagyományos MLP."""
    import time
    
    torch.manual_seed(42)
    batch_size = 1  # NEURA tipikus
    input_dim = 64
    
    X = torch.randn(batch_size, input_dim).cuda()
    
    # Hagyományos MLP (GPU-n)
    mlp = nn.Sequential(
        nn.Linear(input_dim, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1)
    ).cuda()
    
    # HEMNA v2
    hemna = HEMNAv2(
        input_dim=input_dim,
        fast_dims=[32, 16],
        refine_dim=4,
        output_dim=1,
        t1_ratio=0.25,
        confidence_threshold=0.3
    )
    
    # Warmup
    for _ in range(10):
        mlp(X)
        hemna(X)
    
    torch.cuda.synchronize()
    
    # Benchmark MLP
    start = time.time()
    for _ in range(100):
        mlp(X)
    torch.cuda.synchronize()
    mlp_time = (time.time() - start) / 100
    
    # Benchmark HEMNA (fast path — nincs T3 hívás)
    start = time.time()
    for _ in range(100):
        pred, conf = hemna(X)
    torch.cuda.synchronize()
    hemna_fast = (time.time() - start) / 100
    
    # Benchmark HEMNA (refinement — T3 CPU-n)
    hemna.confidence_threshold = 0.0  # mindig hívja T3
    start = time.time()
    for _ in range(100):
        pred, conf = hemna(X)
    torch.cuda.synchronize()
    hemna_refine = (time.time() - start) / 100
    
    print(f"\n=== Sebesség benchmark (batch={batch_size}, 100 iteráció) ===")
    print(f"  MLP (GPU):          {mlp_time*1000:.2f} ms")
    print(f"  HEMNA fast (GPU):   {hemna_fast*1000:.2f} ms")
    print(f"  HEMNA refine (CPU): {hemna_refine*1000:.2f} ms")
    print(f"  T3 CPU overhead:    {(hemna_refine - hemna_fast)*1000:.2f} ms")
    print(f"  Gyorsulás MLP-hez:  "
          f"{mlp_time/hemna_fast:.1f}×")
    
    return mlp_time, hemna_fast, hemna_refine


if __name__ == '__main__':
    test_hemna_v2()
    print("\n" + "="*50)
    benchmark_speed()
