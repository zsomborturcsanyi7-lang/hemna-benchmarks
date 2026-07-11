# HEMNA — Heterogén Multi-Skálájú Neurális Architektúra

**Verzió:** 3.0  
**Szerző:** Zsombi (AI asszisztens segítségével)  
**Státusz:** Kutatás & fejlesztés alatt

---

## Leírás

A HEMNA egy kísérleti neurális architektúra, amely heterogén, többszintű rétegekből épül fel (Tier1, Tier2, Tier3). A projekt célja egy alternatív MLP/FFN architektúra benchmarkolása és összehasonlítása a hagyományos transzformer rétegekkel. A projekt tartalmaz magyar nyelvi modell traininget is (60M → 300M paraméter), SentencePiece tokenizerrel és saját adatpipeline-nal.

---

## Fájlszerkezet

```
HEMNA/
│
├── prototype/                    # Prototípusok és kísérletek
│   ├── bench_all.py              # Teljes HEMNA benchmark (T1+T2+T3 vs MLP)
│   ├── bench_hemna_vs_mlp.py     # HEMNA vs MLP összehasonlítás
│   ├── benchmark_ckpt.py         # Checkpoint benchmark
│   │
│   ├── 300m_terv.md              # 300M magyar modell terv
│   ├── train_300m.py             # 300M paraméteres training
│   ├── continue_300m.py          # Training folytatása
│   ├── train_v2.py               # Továbbfejlesztett training loop
│   │
│   ├── test_100.py               # 100 lépés teszt
│   ├── test_120m.py              # 120M modell teszt
│   ├── test_300m_final.py        # 300M végső teszt
│   ├── test_10steps.py           # 10 lépés gyorsteszt
│   ├── test_minimal.py           # Minimál teszt
│   ├── test_nan.py               # NaN detekció teszt
│   ├── test_dataload.py          # Adatbetöltő teszt
│   ├── test_speed2.py            # Sebesség teszt v2
│   ├── test_accum.py             # Gradiens akkumuláció teszt
│   ├── test_vers.py              # Verzió teszt
│   │
│   ├── combine_data.py           # Adatok kombinálása
│   ├── combine_no_wiki.py        # Kombinálás Wiki nélkül
│   ├── clean_data.py             # Adattisztítás
│   ├── check_data.py             # Adatellenőrzés
│   ├── download_more.py          # További adatok letöltése
│   │
│   ├── extract_subs.py           # Felirat kinyerés
│   ├── check_subs_enc.py         # Felirat kódolás ellenőrzés
│   ├── sample_subs.py            # Felirat mintavételezés
│   ├── tokenize_subs.py          # Felirat tokenizálás
│   │
│   ├── test_lm_cli.py            # Nyelvi modell CLI teszt
│   ├── test_batch.py             # Batch teszt
│   ├── test_batch2.py            # Batch teszt v2
│   ├── profile_step.py           # Profilozó lépés
│   ├── check_pytorch.py          # PyTorch verzió ellenőrzés
│   │
│   ├── read_log.py               # Log olvasó
│   ├── neura_monitor.py          # Training monitor
│   ├── wake_gpu.py               # GPU ébresztő
│   │
│   ├── NEURA_AGENT_PLAN.md       # Agent terv
│   │
│   ├── run_training.bat          # Training indító
│   ├── run_bg.bat                # Háttérben indító
│   ├── run_hold.bat              # Training tartó ablakkal
│   ├── run_training.vbs          # VBS indító
│   ├── create_task.ps1           # Ütemezett feladat (PowerShell)
│   ├── start_schtask.ps1         # Scheduled task indító
│   │
│   ├── _b64.txt ... _b64_v3.txt  # Base64 kódolt adatok
│   └── _b64_working.txt / _b64_opt.txt / _b64_orig.txt
│
└── tests_lm.bat                  # Nyelvi modell tesztek
```

---

## Használat

### HEMNA benchmark futtatása

```bash
cd prototype
pip install torch numpy

# Teljes benchmark (T1+T2+T3 vs MLP)
python bench_all.py

# HEMNA vs MLP összehasonlítás
python bench_hemna_vs_mlp.py
```

### Magyar nyelvi modell training

```bash
# 300M paraméteres modell training
cd prototype
python train_300m.py

# VAGY batch fájllal
run_training.bat

# Training folytatása
python continue_300m.py

# Monitorozás
python neura_monitor.py
```

### Adatelőkészítés

```bash
# Feliratok kinyerése
python extract_subs.py

# Tokenizálás
python tokenize_subs.py

# Adatok kombinálása
python combine_data.py

# Adattisztítás
python clean_data.py
```

---

## 300M Modell Terv

| Preset | Dim | Layers | Heads | FFN | Paraméterek | Batch | Seq Len |
|--------|-----|--------|-------|-----|-------------|-------|---------|
| 60M | 768 | 6 | 12 | 2304 | 60M | 4 | 512 |
| **120M** | 768 | 12 | 12 | 2304 | 120M | 4 | 512 |
| 180M | 1024 | 12 | 16 | 3072 | 180M | 4 | 256 |
| **300M** | 1024 | 24 | 16 | 3072 | 354M | 2 | 256 |

### Architektúra
- Decoder-only Transformer
- GQA (Grouped Query Attention) — 16 head, 4 KV head
- Gated FFN (SwiGLU)
- RMSNorm normalizáció
- Tied embedding (input és output közös)

---

## Függőségek

- **Python** 3.10+
- **PyTorch** 2.0+ (CUDA ajánlott)
- **SentencePiece** (tokenizer)
- **NumPy**

---

## Fejlesztő

Zsombi (AI asszisztens segítségével) (AI asszisztens segítségével)
