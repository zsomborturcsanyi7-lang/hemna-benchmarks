# NEURA 300M → Intelligens Beszélgető Partner + Tool Agent

## Terv: 0-ról működő magyar nyelvű kiságensre az RTX 3070-en

---

## 1. REALITÁSOK — Mit tud és mit NEM tud egy 354M-es modell?

### Amit a kutatás mond:

| Forrás | Modell | Eredmény |
|--------|--------|----------|
| **OPT-350M SFT** (AAAI 2026) | 350M | **77.55% pass rate ToolBench** — jobb, mint GPT-3.5 (26%) és ToolLLaMA-7B (30%) |
| **TinyAgent** (EMNLP 2024) | 1.1B | Felülmúlja GPT-4-Turbo-t specifikus function calling taskokon |
| **4-bit LoRA SLM** | 350M-1.5B | 10% → 79% syntactic validity function callingban |
| **MobileLLM** (ICML 2024) | 125M-350M | SwiGLU + GQA + Tied Embeddings = pont a NEURA architektúrája |

### Következtetés:
**IGEN, lehet belőle tool-use agent.** De:
- Nem lesz "általános intelligencia" — egy **specializált intent-parser + tool-caller** lesz
- A minőséget a **fine-tuning adat minősége** határozza meg, nem a modell mérete
- **NEM** tud hosszú több-lépéses reasoninget — a komplex logikát a toolokba kell tenni

---

## 2. JELENLEGI KORLÁTOK (NEURA 300M)

| Korlát | Hatás | Megoldás |
|--------|-------|----------|
| PPL ~100-200 | Sok hibát generál, értelmetlen szövegek | Több pre-training, majd SFT |
| Csak 2.53B tokenen tanult | Modern modellek 10-100T+ token | Részben áthidalható SFT-vel + disztillációval |
| SentencePiece 32K vocab | Nincs token a JSON/{/}/(/) számokhoz | **Vocab bővítés kell** |
| Nincs instruction tuning | Nem követ utasításokat | SFT teljesen magyar adaton |
| FP32, 1.45 GB | Lassú inference | Quantization (FP16/INT8/GGUF) |
| RTX 3070 8GB VRAM | 354M FP32 kényelmesen befér, de batch limitált | FP16 → 700MB, INT8 → 350MB |
| Kontextus ablak ~512 token | Túl rövid beszélgetéshez | RoPE scaling → 2048+ token |
| Csak magyar nyelv | Tool-ök nevei angolok | Kevert nyelvű SFT adat kell |

---

## 3. FÁZISOK — A TELJES ÚT

```

┌─────────────────────────────────────────────────────────────┐
│ FÁZIS 0: Pre-training befejezése                            │
│  → 230K lépés (már fut, ~reggelre kész)                     │
│  → checkpoint lm300m_v2_step230000.pt                       │
├─────────────────────────────────────────────────────────────┤
│ FÁZIS 1: Vocab bővítés + Context extension                  │
│  → Új tokenek: JSON speciális karakterek, számok            │
│  → RoPE scaling 512→2048                                    │
│  → Rövid folytató pre-training (1000 step)                  │
├─────────────────────────────────────────────────────────────┤
│ FÁZIS 2: SFT — Instruction Following                        │
│  → Magyar instruct adat (PULI 44K + saját)                  │
│  → Chat formátum tanítása (user/assistant/system)           │
│  → LoRA vagy full fine-tune (full jobb, de drágább)         │
├─────────────────────────────────────────────────────────────┤
│ FÁZIS 3: Tool-use fine-tuning                               │
│  → Function calling JSON formátum                           │
│  → Hungarian tool-leírásokkal + példákkal                   │
│  → Multi-turn tool hívások (tool call → result → response)  │
├─────────────────────────────────────────────────────────────┤
│ FÁZIS 4: Knowledge Distillation (ha kell)                   │
│  → Teacher: Hermes Pro / Claude / GPT (API)                 │
│  → Student: NEURA 300M                                      │
│  → Kimeneti logit-ok + generált adat tanítása               │
├─────────────────────────────────────────────────────────────┤
│ FÁZIS 5: Inference optimalizálás + Deployment               │
│  → FP16 / INT4 quantization                                 │
│  → KV-cache optimalizálás                                   │
│  → API szerver (OpenAI kompatibilis)                        │
│  → Tool orchestrator integráció                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. RÉSZLETES TERV

### FÁZIS 0: Pre-training befejezése (most)

- **Jelenleg fut:** Step 17000/100000 (total 157000), ~549 tok/s
- **Cél:** 100K további lépés a 140K checkpointból
- **Eredmény:** `lm300m_v2_step240000.pt` (total ~240K)
- **Idő:** ~10 óra → holnap reggelre kész
- **PPL cél:** 50-100 közötti validation PPL

### FÁZIS 1: Vocab bővítés + Context Extension

Miért kell:
- A SentencePiece 32K tokenizátor NEM tartalmaz külön tokeneket JSON struktúrákhoz
- Pl. `{`, `}`, `[`, `]`, `"`, `:`, `,` — ezek nélkül nem lehet tool call-t generálni
- Számok: `0-9`, `.`, `-` — fontos tool argumentumokhoz
- Jelenlegi kontextus 512 token — túl rövid

Megoldás:
```python
# 1. Tokenizer bővítés új tokenekkel
new_tokens = [
    "<TOOL_CALL>", "<TOOL_RESULT>",
    "{", "}", "[", "]", ":", ",", "\"",
    "<|im_start|>", "<|im_end|>",
    "0","1","2","3","4","5","6","7","8","9",
    ".",
]
spm.AddSpecialTokens(new_tokens)

# 2. Embedding mátrix bővítés
old_emb = model.tok.weight  # [32000, dim]
new_emb = torch.cat([old_emb, torch.randn(len(new_tokens), dim) * 0.02])
model.tok = nn.Embedding(V + len(new_tokens), dim)
model.tok.weight.data = new_emb

# 3. Output layer bővítés (ha nincs tied embedding)
new_out = torch.cat([model.out.weight, torch.randn(len(new_tokens), dim) * 0.02])
model.out = nn.Linear(dim, V + len(new_tokens), False)
model.out.weight.data = new_out

# 4. RoPE scaling 512→2048
# Beépített scaling: a mask méretének növelése + interpoláció
# Egyszerű mód: self.register_buffer('m', torch.tril(torch.ones(2048, 2048)))
```

Utána: **Rövid folytató pre-training** (500-1000 step) hogy az új token embeddingek beépüljenek. Ehhez elég a meglévő adat, amin átpasszoljuk a tokenizert.

### FÁZIS 2: SFT — Instruction Following

**Adatforrások:**

| Forrás | Méret | Leírás |
|--------|-------|--------|
| **PULI magyar instruct** (ELTE) | 44,626 | Magyar instruction-following példák |
| **Stanford Alpaca** (magyarra fordítva) | 52K | Általános utasításkövetés |
| **OpenAssistant** (magyar rész) | ~10K | Többfordulós beszélgetések |
| **Saját generált** (Hermes Pro-ból) | 10-50K | API-val generált magyar chat adat |

**Formátum:**
```
<|im_start|>system
Te egy segítőkész magyar asszisztens vagy. Válaszolj a felhasználó kérdéseire.
<|im_end|>
<|im_start|>user
Mi a magyar főváros?
<|im_end|>
<|im_start|>assistant
Magyarország fővárosa Budapest, amely a Duna két partján fekszik. Lakossága körülbelül 1.7 millió fő.
<|im_end|>
```

**Training:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from peft import LoraConfig, get_peft_model

# LoRA config (ha full fine-tune nem fér ki)
lora_config = LoraConfig(
    r=64, lora_alpha=128,
    target_modules=["wq", "wk", "wv", "wo", "w1", "w2", "w3"],
    lora_dropout=0.1,
)

# VAGY full fine-tune (jobb minőség, de több VRAM)
# BS=1, FP16, gradient accumulation=4 → kb 6GB VRAM
```

**Kimenet:** `neura-300m-instruct` — képes chat formátumban válaszolni.

### FÁZIS 3: Tool-use fine-tuning

**Tool-leírások formátuma (a modellnek adott system prompt):**
```
<|im_start|>system
Az alábbi függvények állnak rendelkezésre:

függvény: get_idojaras(varos: str) -> dict
  Leírás: Visszaadja az aktuális időjárást egy városban
  Paraméterek:
    varos (str): A város neve

függvény: search_web(kereses: str) -> list
  Leírás: Webes keresést végez
  Paraméterek:
    keresés (str): A keresési kifejezés

függvény: send_email(cimzett: str, tema: str, tartalom: str) -> bool
  Leírás: Email küldése
  Paraméterek:
    cimzett (str): Az email cím
    tema (str): Az email tárgya
    tartalom (str): Az email tartalma

Amikor egy függvényt kell meghívnod, használd a következő formátumot:
<TOOL_CALL>
{"name": "függvény_neve", "arguments": {"param1": "érték1"}}
</TOOL_CALL>
<|im_end|>
```

**Training adat formátum (function calling):**
```
<|im_start|>user
Mi az időjárás Budapesten?
<|im_end|>
<|im_start|>assistant
<TOOL_CALL>
{"name": "get_idojaras", "arguments": {"varos": "Budapest"}}
</TOOL_CALL>
<|im_end|>
<|im_start|>tool
{"homerseklet": 22, "leiras": "Napos", "paratartalom": 45}
<|im_end|>
<|im_start|>assistant
Budapesten jelenleg 22°C van, napos idővel. A páratartalom 45%.
<|im_end|>
```

**Adat generálása:**
- Használj Hermes Pro / Claude / GPT API-t hogy generáljon 1000-5000 tool-use példát magyar nyelven
- Minden példa: user kérés → tool call → tool result → assistant válasz
- Változatos tool kombinációk (1 tool, 2 tool egymás után, 0 tool)

**Training config:**
- Full fine-tune (kis modell, belefér)
- 3-5 epoch
- Learning rate: 2e-5
- BS=1, grad accumulation

**Várható eredmény:**
- OPT-350M alapján: **~70-77% tool call accuracy**
- A többi esetben a modell szövegesen válaszol (fallback)

### FÁZIS 4: Knowledge Distillation (Ha kell)

Ha az SFT után a modell még mindig gyenge:

**Black-box distillation** (API-ból):
```python
# 1. Teacher (GPT/Claude/Hermes Pro) generál válaszokat
teacher_response = call_llm_api(prompt=user_query, tools=tool_defs)

# 2. Student (NEURA) tanul ezekből
train_on_examples(teacher_data)
```

**White-box distillation** (saját nagymodellből):
```python
# 1. Teacher forward pass → logits
teacher_logits = teacher_model(input_ids)

# 2. Student forward pass
student_logits = student_model(input_ids)

# 3. KL divergence loss a kettő között
loss = kl_divergence(teacher_logits, student_logits)
```

A TAID módszer (Sakana AI) ezt hatékonyan csinálja.

### FÁZIS 5: Inference + Deployment

**1. Quantization:**

| Módszer | Méret | Sebesség | Minőség |
|---------|-------|----------|---------|
| FP32 (most) | 1.45 GB | 100% | referenciaként |
| FP16 | 725 MB | ~120% | ~0% veszteség |
| INT8 (GPTQ) | 360 MB | ~140% | ~1-2% veszteség |
| INT4 (GGUF Q4_K_M) | ~200 MB | ~150% | ~3-5% veszteség |

RTX 3070-es javaslat: **FP16** — gyors, pontosan annyi amennyi, és 700 MB pont belefér a tool use overhead mellé.

**2. Inference server:**
```python
# OpenAI-kompatibilis API szerver
from flask import Flask, request, jsonify

@app.route("/v1/chat/completions", methods=["POST"])
def chat():
    messages = request.json["messages"]
    tools = request.json.get("tools", [])
    
    # Formázás a modell számára
    prompt = format_chat_template(messages, tools)
    
    # Generálás
    output = model.generate(prompt, max_tokens=512)
    
    # Tool call kinyerése
    tool_call = extract_tool_call(output)
    
    return jsonify({
        "choices": [{
            "message": {
                "content": output if not tool_call else None,
                "tool_calls": tool_call
            }
        }]
    })
```

**3. Tool orchestrator (a modell körül):**

A modell NEM maga hívja meg a toolokat — a modell KIIRJA a tool call JSON-t, és egy külső orchestrator:
1. Parsolja a JSON-t
2. Meghívja a tool-t
3. Visszaadja az eredményt a modellnek
4. A modell megválaszolja az eredményt

```
Felhasználó → [Orchestrator] → NEURA 300M (kiírja a tool call-t)
                                   ↓
                              Orchestrator parsolja
                                   ↓
                              Tool végrehajtás
                                   ↓
                              Eredmény → NEURA 300M (válasz)
                                   ↓
                              Válasz a felhasználónak
```

**4. Sebességbecslés:**

| Művelet | FP32 | FP16 |
|---------|------|------|
| Prefill (első token) | ~20ms | ~12ms |
| Generálás/token | ~8ms | ~5ms |
| 100 token válasz | ~820ms | ~512ms |
| Teljes round trip (1 tool) | ~2s | ~1.2s |

→ **Használható!** Nem villámgyors, de chat-re alkalmas.

**5. Memória:**
```
FP16 modell:         725 MB
KV cache (2048 ctx):  ~80 MB
Tool JSON buffer:     ~10 MB
Egyéb overhead:      ~100 MB
Összesen:           ~915 MB ← bőven elfér 8GB-ból
```

---

## 5. MIT LEHET MEGTANÍTANI A MODELLNEK (Tool-ök)

### Javasolt tool készlet (kezdetben 5-10 tool):

| Tool | Leírás | Bonyolultság |
|------|--------|-------------|
| `get_idojaras(varos)` | Időjárás lekérdezés | Könnyű |
| `search_web(kereses)` | Webes keresés | Könnyű |
| `szamolo(muvelet, szamok)` | Matematikai műveletek | Könnyű |
| `fordito(szoveg, nyelv)` | Fordítás | Közepes |
| `get_aktualis_ido()` | Aktuális idő/dátum | Könnyű |
| `send_email(cimzett, tema, tartalom)` | Email küldés | Közepes |
| `get_hírek(tema, db)` | Hírek lekérdezése | Könnyű |
| `wikipedia_osszefoglalo(szocikk)` | Wikipedia összefoglaló | Könnyű |
| `emlekezteto(mettol, meddig, szoveg)` | Emlékeztető beállítás | Közepes |

### Amit NE vegyünk bele (túl nehéz 354M-nek):
- Több-lépéses komplex workflow-k (pl. "foglalj repjegyet, szállást, és biztosítást")
- Kódgenerálás (külön modell kellene)
- SQL lekérdezések
- Multi-tool párhuzamos hívás (LLMCompiler-t használhatjuk, de az külső)

---

## 6. MAGYAR ADAT STRATÉGIA

### Források magyar instruction/data:

1. **PULI magyar instruct adat (44K)** — ingyenes, ELTE kutatóközpont
2. **Alpaca magyarra fordítva** — Google Translate API, ~$50
3. **OpenAssistant magyar szálak** — nyílt forrás
4. **Saját generált Hermes Pro-ból**:
   ```
   Rendszer: "Generálj egy magyar instruction-válasz párt..."
   → 10,000 példa ~$5-10 API költség
   ```
5. **Tool-use szintetikus adat**:
   ```
   Rendszer: "Generálj egy tool-hívás példát magyar nyelven..."
   → 5,000 példa ~$5-10 API költség
   ```

**Teljes becsült adatméret:** 50-100K példa

---

## 7. MIT VÁRHATUNK REÁLISAN (best case)

| Képesség | Elvárás |
|----------|---------|
| Egyszerű kérdések megválaszolása | ✅ 70-80% jó |
| Tool hívás (1 lépés) | ✅ 70-77% (OPT-350M alapján) |
| Többfordulós chat (3-5 kör) | ⚠️ 50-60% koherens |
| Multi-step tool használat | ❌ 30-40% |
| Hosszú kontextus (>1024 token) | ❌ Felejt |
| Összetett reasoning | ❌ Nem erre való |
| **Használható tool agent** | **✅ IGEN, de egyszerű feladatokra** |

---

## 8. MENETREND ÉS IDŐBECSLÉS

| Fázis | Idő | Költség (API) |
|-------|-----|---------------|
| **F0:** Pre-training kész | ~10 óra (ma éjjel) | $0 (RTX 3070) |
| **F1:** Vocab bővítés + context | ~30 perc munka + 1 óra training | $0 |
| **F2:** SFT adat előkészítés | ~2-3 óra | ~$10-20 (fordítás/generálás) |
| **F2:** SFT training | ~2-4 óra RTX 3070 | $0 |
| **F3:** Tool adat generálás | ~2 óra | ~$10-20 (API) |
| **F3:** Tool fine-tune | ~2-3 óra RTX 3070 | $0 |
| **F4:** Disztilláció (ha kell) | 3-5 óra API + 2 óra train | ~$30-50 |
| **F5:** Quantizáció + server | ~1 óra | $0 |
| **ÖSSZESEN** | **~15-20 óra munka** | **~$50-90** |

---

## 9. A LEGNAGYOBB KOCKÁZATOK

1. ⚠️ **A modell túl kicsi** — 354M tényleg a határon van. A PPL-nek 50 ALÁ kell mennie mielőtt SFT-nek van értelme.
2. ⚠️ **Magyar adat hiány** — Nincs elég magas minőségű magyar tool-use dataset. A generált adat minősége kérdéses.
3. ⚠️ **Tokenizátor limit** — SentencePiece bővítése kockázatos, ronthatja a meglévő tudást.
4. ⚠️ **Hallucináció** — Kis modellek többet hallucinálnak. A tool-okba épített validáció kritikus.
5. ⚠️ **Kontextus korlát** — Még RoPE scalinggel sem lesz >2048 token a kontextus.

---

## 10. JAVASOLT ELSŐ LÉPÉS (ha egyetértek)

1. **Ma éjjel:** Hagyjuk a traininget futni
2. **Holnap reggel:** Ellenőrizzük a PPL-t — ha **<50**, van értelme továbbmenni
3. **Holnap:** PULI adat letöltése + magyar Alpaca fordítás indítása
4. **Holnap délután:** SFT kipróbálása 1 epoch-al → chat teszt
5. **Ha a chat működik:** Tool-adat generálás → tool fine-tune → deployment

Ha a PPL **>100** marad a 240K lépés után is, akkor a modell egyszerűen nem tanult eleget a pre-training során, és érdemes lehet előbb:
- Több adat (még több magyar korpusz)
- Hosszabb pre-training (még 100K lépés)
- Vagy elfogadni, hogy csak korlátozottan használható
