# HEMNA — Heterogeneous Multi-Scale Neural Architecture

**Version:** 3.0  
**Author:** Zsombi & Hermes Agent (Nous Research)  
**Status:** Research & development

---

## Description

HEMNA is an experimental neural architecture built from heterogeneous, multi-level layers (Tier1, Tier2, Tier3). The project aims to benchmark and compare an alternative MLP/FFN architecture against traditional transformer layers. The project also includes Hungarian language model training (60M → 300M parameters), with a SentencePiece tokenizer and a custom data pipeline.

---

## File Structure

```
HEMNA/
│
├── prototype/                    # Prototypes and experiments
│   ├── bench_all.py              # Full HEMNA benchmark (T1+T2+T3 vs MLP)
│   ├── bench_hemna_vs_mlp.py     # HEMNA vs MLP comparison
│   ├── benchmark_ckpt.py         # Checkpoint benchmark
│   │
│   ├── 300m_terv.md              # 300M Hungarian model plan
│   ├── train_300m.py             # 300M parameter training
│   ├── continue_300m.py          # Continue training
│   ├── train_v2.py               # Improved training loop
│   │
│   ├── test_100.py               # 100-step test
│   ├── test_120m.py              # 120M model test
│   ├── test_300m_final.py        # 300M final test
│   ├── test_10steps.py           # 10-step quick test
│   ├── test_minimal.py           # Minimal test
│   ├── test_nan.py               # NaN detection test
│   ├── test_dataload.py          # Data loader test
│   ├── test_speed2.py            # Speed test v2
│   ├── test_accum.py             # Gradient accumulation test
│   ├── test_vers.py              # Version test
│   │
│   ├── combine_data.py           # Data combination
│   ├── combine_no_wiki.py        # Combine without Wiki
│   ├── clean_data.py             # Data cleaning
│   ├── check_data.py             # Data verification
│   ├── download_more.py          # Download additional data
│   │
│   ├── extract_subs.py           # Subtitle extraction
│   ├── check_subs_enc.py         # Subtitle encoding check
│   ├── sample_subs.py            # Subtitle sampling
│   ├── tokenize_subs.py          # Subtitle tokenization
│   │
│   ├── test_lm_cli.py            # Language model CLI test
│   ├── test_batch.py             # Batch test
│   ├── test_batch2.py            # Batch test v2
│   ├── profile_step.py           # Profiler step
│   ├── check_pytorch.py          # PyTorch version check
│   │
│   ├── read_log.py               # Log reader
│   ├── neura_monitor.py          # Training monitor
│   ├── wake_gpu.py               # GPU wake-up
│   │
│   ├── NEURA_AGENT_PLAN.md       # Agent plan
│   │
│   ├── run_training.bat          # Training launcher
│   ├── run_bg.bat                # Background launcher
│   ├── run_hold.bat              # Training with persistent window
│   ├── run_training.vbs          # VBS launcher
│   ├── create_task.ps1           # Scheduled task (PowerShell)
│   ├── start_schtask.ps1         # Scheduled task launcher
│   │
│   ├── _b64.txt ... _b64_v3.txt  # Base64-encoded data
│   └── _b64_working.txt / _b64_opt.txt / _b64_orig.txt
│
└── tests_lm.bat                  # Language model tests
```

---

## Usage

### Running HEMNA Benchmark

```bash
cd prototype
pip install torch numpy

# Full benchmark (T1+T2+T3 vs MLP)
python bench_all.py

# HEMNA vs MLP comparison
python bench_hemna_vs_mlp.py
```

### Hungarian Language Model Training

```bash
# 300M parameter model training
cd prototype
python train_300m.py

# OR via batch file
run_training.bat

# Continue training
python continue_300m.py

# Monitoring
python neura_monitor.py
```

### Data Preparation

```bash
# Subtitle extraction
python extract_subs.py

# Tokenization
python tokenize_subs.py

# Data combination
python combine_data.py

# Data cleaning
python clean_data.py
```

---

## 300M Model Plan

| Preset | Dim | Layers | Heads | FFN | Parameters | Batch | Seq Len |
|--------|-----|--------|-------|-----|------------|-------|---------|
| 60M | 768 | 6 | 12 | 2304 | 60M | 4 | 512 |
| **120M** | 768 | 12 | 12 | 2304 | 120M | 4 | 512 |
| 180M | 1024 | 12 | 16 | 3072 | 180M | 4 | 256 |
| **300M** | 1024 | 24 | 16 | 3072 | 354M | 2 | 256 |

### Architecture
- Decoder-only Transformer
- GQA (Grouped Query Attention) — 16 heads, 4 KV heads
- Gated FFN (SwiGLU)
- RMSNorm normalization
- Tied embedding (shared input and output)

---

## Dependencies

- **Python** 3.10+
- **PyTorch** 2.0+ (CUDA recommended)
- **SentencePiece** (tokenizer)
- **NumPy**

---

## Developer

Zsombi & Hermes Agent (Nous Research)
