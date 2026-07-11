# 300M magyar nyelvi modell - Terv
# ================================
# Cel: 300M+ parameteru nyelvi modell a NEURA 4 shard adatan (156M token)

## Architektura
- **Tipus**: Decoder-only Transformer
- **Attention**: GQA (Grouped Query Attention) - 16 head, 4 KV head
- **FFN**: Gated FFN (SwiGLU)
- **Normalizacio**: RMSNorm
- **Embedding**: Tied (input es output kozos)
- **Pozicio**: Nincs (ALiBi helyett implicit pozicio a causal mask-bol)

## Konfiguraciok

| Preset  | Dim  | Layers | Heads | FFN   | Params | Batch | Seq  | CKPT | TDR? |
|---------|------|--------|-------|-------|--------|-------|------|------|------|
| 60M     | 768  | 6      | 12    | 2304  | 60M    | 4     | 512  | Nem  | Nem  |
| **120M**| 768  | 12     | 12    | 2304  | **120M**| 4   | 512  | Igen | Nem  |
| 180M    | 1024 | 12     | 16    | 3072  | 180M   | 4     | 256  | Igen | Lehet|
| 300M*   | 1024 | 24     | 16    | 3072  | 354M   | 2     | 256  | Igen | Igen |

## TDR (Timeout Detection and Recovery) megoldasok
A Windows GPU driver 2 masodperc utan TDR-t indit, ha egy CUDA kernel 
tul sokaig fut. Ez a 180M+ modelleknel elofordulhat.

**Megoldasok:**
1. **Gradient checkpointing** ✅ - Beepitve a train_v2.py-ba
2. **Seq len csokkentes** 512 -> 256 - Felezi a forward/backward idot
3. **Batch meret csokkentes** 4 -> 2 - Felezi a VRAM-ot es az idot
4. **Windows TDR registry modositas** (TdrDelay=8) - Nincs SSH hozzaferes
5. **torch.compile** - Kisebb CUDA kernelek, de elso lepes lassabb

## Training terv
1. **60M** (fut) - 50K lepes, ~1.9 ora ✅ (elerheto: PPL ~5-10)
2. **120M** (kovetkezo) - 50K lepes, ~3 ora (PPL varhato: 3-5)
3. **180M** (ha 120M jo) - 50K lepes, ~4 ora (PPL varhato: 2-4)
4. **300M** (vegs cel) - 50K lepes, ~6 ora (PPL varhato: 1.5-3)

## Elemzes
Miutan a 60M befejezodott, ellenorizni:
1. Loss/PPL trend az utolso 10K lepesben
2. Szoveg generalas minosege
3. Overfitting: train vs val loss kulonbseg
4. Memoria hasznalat (csak a 60M elfutott-e 8GB-ban)

Ha PPL < 15 es nincs overfitting, a 120M futtathato.
Ha PPL > 15, tobb lepes kell (100K) vagy tobb adat.

## 300M modell futtatasa
A 300M modell TDR-t okozhat. A kovetkezo optimalizaciokkal:
- bs=2, seq=256, gradient checkpointing
- Windows TDR novelese (ha lehet SSH-n keresztul)
- Alternativ: 2x 150M modell ensemble helyett 1x 300M
