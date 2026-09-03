# VisionFlow

Medical images can't go to the cloud. Surgical reports contain PHI. Supply chain manifests are proprietary. Every enterprise team with vision-language needs is stuck choosing between privacy and capability. VisionFlow closes that gap — a fully local, quantized Vision-Language Model pipeline that runs on a MacBook and returns structured data, not a cloud API bill.

VisionFlow loads a small VLM ([SmolVLM](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct)), auto-detects your hardware *and picks a model size that fits it*, runs inference entirely on-device, and constrains the decoder so the output is valid JSON by construction rather than by hope.

**No image, prompt, or extracted field ever leaves the machine.**

## Install

```bash
pip install visionflow-edge
```

The distribution is `visionflow-edge`; the import and CLI are plain `visionflow` / `vf`. (The bare `visionflow` name on PyPI is a registered-but-empty project owned by someone else.)

Or from source:

```bash
git clone https://github.com/vineetha00/visionflow && cd visionflow && pip install -e .
```

Optional extras: `[cuda]` for bitsandbytes INT4/INT8, `[cpu]` for the llama.cpp GGUF path, `[onnx]` / `[onnx-gpu]` for ONNX Runtime, `[constrained]` for full-vocabulary constrained decoding via `outlines`, `[eval]` for the GPT-4o baseline.

## Quickstart

All three output modes, in under 10 lines:

```python
from visionflow import VisionFlow

vf = VisionFlow()  # auto-detects CUDA / MPS / CPU, and sizes the model to the device
vf.load()

vf.text("document.png", "What kind of document is this?")                          # free-form text
vf.json("document.png", "Extract the patient data", schema={"mrn": "string"})       # structured JSON
vf.key_value("document.png", "Extract shipment info", fields=["po_number", "qty"])  # key-value
```

Guarantee valid JSON at the decoder instead of repairing it afterwards:

```python
vf.json("document.png", "Extract the patient data", schema={"mrn": "string"}, constrained=True)
```

From the command line:

```bash
visionflow document.png --prompt "Extract the invoice fields" --mode json
```

```bash
vf bench --repeats 3
```

```bash
vf accuracy
```

## Architecture

```
                    ┌──────────────────────┐
   image + prompt → │   1. Model Engine     │  hardware auto-detect (CUDA/MPS/CPU)
                    │   (engine.py)         │  → quantization path + model size tier
                    └──────────┬────────────┘
                               │ raw generated text
                    ┌──────────▼────────────┐
                    │ 2. VisionFlow Pipeline│  wraps prompt with output-mode
                    │   (pipeline.py)       │  instructions (text / json / kv)
                    └──────────┬────────────┘
                               │
                    ┌──────────▼────────────┐         ┌──────────────────────┐
                    │ 3. Extraction Layer   │◀────────│  constrained.py       │
                    │   (extractors.py)     │  masks  │  JSON grammar at the  │
                    │   parse + repair pass │  tokens │  decoder (opt-in)     │
                    └──────────┬────────────┘         └──────────────────────┘
                               │
                        structured output
                     (dict / list / plain text)

   bench.py  — latency / throughput / memory per backend, in isolated processes
   accuracy.py — schema validity + field accuracy per quantization level
```

## Use cases

### Medical document intelligence

Input: a scanned surgical report. Output: structured JSON with patient data, procedure details, diagnosis code, and flagged anomalies — extracted entirely on-device, no PHI transmitted anywhere.

```bash
python examples/medical_document.py
```

![Medical document extraction demo](docs/demo_medical.gif)

### Supply chain document processing

Input: a cold-chain shipping manifest. Output: structured JSON/key-value with manifest header fields and line-item product codes, quantities, lots, and expiry dates.

```bash
python examples/supply_chain_manifest.py
```

### Chart and dashboard understanding

Input: a screenshot of a chart. Output: extracted data points and a trend description, structured for downstream use.

```bash
python examples/chart_extraction.py
```

![Chart extraction demo](docs/demo_chart.gif)

All three examples run against synthetic, fabricated sample images in `examples/sample_images/` — none of this is real patient or shipment data.

## Hardware compatibility and capability tiers

Model size is part of hardware detection, not a constant. A 2.25B VLM is unusable on a Raspberry Pi even at INT4, so `VisionFlow()` picks a tier that fits the device and the benchmark table below makes the accuracy cost of that choice visible.

| Device | Model tier selected | Quantization path | Status |
|---|---|---|---|
| NVIDIA GPU, ≥6GB VRAM | SmolVLM-2.25B | `bitsandbytes` INT4 (NF4), INT8 fallback | ✅ Measured on an L40S — INT4 runs in 1.85GB VRAM at 47.7 tok/s. **Prefer INT4 over the INT8 fallback**: INT8 is slower *and* less accurate here |
| NVIDIA GPU, 3–6GB VRAM | SmolVLM-500M | `bitsandbytes` INT4 | ⚠️ Implemented, not benchmarked |
| Apple Silicon, ≥16GB | SmolVLM-2.25B | fp16 on `mps` | ✅ Measured below — **fast, but see the fp16 accuracy finding** |
| Apple Silicon, 8–16GB | SmolVLM-500M | fp16 on `mps` | ✅ Path measured at the 256M tier below |
| CPU / Raspberry Pi, <4GB | SmolVLM-256M | fp32, or GGUF INT4 via `llama-cpp-python` | ✅ fp32 measured below; GGUF implemented, not benchmarked |
| CPU, ≥4GB | SmolVLM-500M | fp32 or GGUF | ✅ Measured below at both 2.25B and 256M |

Override the automatic choice at any time: `VisionFlow(model_id="HuggingFaceTB/SmolVLM-256M-Instruct")`.

## Benchmarking methodology

`vf bench` is the part of this repo I'd point at first. Five decisions make its numbers trustworthy:

1. **Every configuration runs in a fresh subprocess.** This is not cosmetic. An earlier version of this repo measured MPS and then CPU in one process and reported 15,315.8 MB and 15,313.8 MB peak memory for the two tiers — numbers that differ by 0.01% because the CPU row was reading the MPS driver's still-live allocations from the previous tier. Isolated, the same two configurations report what actually happened: the CPU tier allocates *nothing* on any device and holds 7.4GB of process RSS, while the MPS tier holds 15.3GB of device memory and only 335MB of RSS. Peak memory means nothing if something else already allocated on the device.
2. **Warmup runs are excluded.** The first generation pays lazy kernel and shader compilation. That cost is real, so it's reported separately as model load time rather than being smeared across steady-state latency.
3. **Tokens/sec is reported next to latency.** SmolVLM's output length varies several-fold by image category, so seconds-per-sample alone conflates "this backend is slow" with "this image produced more tokens."
4. **p50 and p95, not just mean ± std.** These latency distributions are strongly right-skewed; a mean ± std implies a symmetry that isn't there.
5. **Unavailable configurations are printed as explicit skips with reasons.** Omitting a row reads as "not applicable"; a skip row reads as "nobody measured this."

One caveat the table can't fix: **peak RSS is a lower bound**, not the checkpoint size. Safetensors weights are memory-mapped, so pages the kernel never faults in don't count toward RSS. It answers "how much did this process keep resident," which is the number that decides whether a device swaps — but it will read lower than the model's size on disk. The device-memory column comes from the backend's own allocator and doesn't have this problem, but only exists for MPS and CUDA.

Reproduce with:

```bash
vf bench --repeats 3 --max-new-tokens 256
```

### Latency, throughput, and memory

| Configuration | Model | Quant | Load (s) | p50 (s) | p95 (s) | mean ± std (s) | tok/s | Peak RSS (MB) | Device mem (MB) |
|---|---|---|---|---|---|---|---|---|---|
| CUDA / INT4 (bnb nf4) | SmolVLM-Instruct | — | — | — | — | *not measured — no CUDA device on this machine* | — | — | — |
| CUDA / INT8 (bnb) | SmolVLM-Instruct | — | — | — | — | *not measured — no CUDA device on this machine* | — | — | — |
| Apple MPS / fp16 | SmolVLM-Instruct | fp16-mps | 8.71 | 27.39 | 39.53 | 26.94 ± 8.55 | 6.62 | 335 | 15,316 |
| CPU / fp32 | SmolVLM-Instruct | fp32-cpu | 9.06 | 152.19 | 204.13 | 144.02 ± 48.83 | 0.90 | 7,435 | — |
| CPU / fp32 (SmolVLM-256M) | SmolVLM-256M-Instruct | fp32-cpu | 3.39 | 10.82 | 12.42 | 10.34 ± 1.68 | 14.83 | 2,850 | — |
| Apple MPS / fp16 (SmolVLM-256M) | SmolVLM-256M-Instruct | fp16-mps | 3.65 | 7.21 | 8.77 | 6.53 ± 2.07 | 22.82 | 1,005 | 5,668 |

Host: macOS 15.1, Apple M3, 16GB RAM, torch 2.13.0. 3 images × 3 repeats, 1 warmup run excluded, `max_new_tokens=256`, each configuration in an isolated subprocess. Raw output: [`benchmarks/results/bench_report.json`](benchmarks/results/bench_report.json).

#### NVIDIA L40S (USC CARC)

Same harness, same images, on an L40S (Ada, 48GB, driver 580.159.04), torch 2.13.0+cu126, bitsandbytes 0.50.2:

| Configuration | Model | Quant | Load (s) | p50 (s) | p95 (s) | mean ± std (s) | tok/s | Peak RSS (MB) | Device mem (MB) |
|---|---|---|---|---|---|---|---|---|---|
| CUDA / INT4 (bnb nf4) | SmolVLM-Instruct | int4-bnb-nf4 | 23.65 | **1.51** | 1.56 | 1.37 ± 0.25 | **47.73** | 2,007 | **1,854** |
| CUDA / INT8 (bnb) | SmolVLM-Instruct | int8-bnb | 15.11 | 5.13 | 5.67 | 5.01 ± 0.62 | 17.82 | 3,581 | 3,009 |
| CPU / fp32 | SmolVLM-Instruct | fp32-cpu | 19.82 | 43.42 | 79.58 | 53.19 ± 19.72 | 2.44 | 10,676 | — |
| CPU / fp32 (SmolVLM-256M) | SmolVLM-256M-Instruct | fp32-cpu | 15.30 | 13.98 | 16.29 | 12.75 ± 3.71 | 12.03 | 2,575 | — |

- **INT4 is 3.7× faster than INT8** (1.37s vs 5.01s; 47.73 vs 17.82 tok/s) and uses 1.6× less device memory. That ordering is backwards from the intuition that fewer bits means more reconstruction work — `bitsandbytes`' LLM.int8() path carries a well-known dequantization overhead, and here it costs more than the precision is worth. See the accuracy table: INT8 doesn't buy accuracy either.
- **INT4 on an L40S is 19.7× faster per sample than fp16 on the M3** (1.37s vs 26.94s) and 7.2× on tokens/sec, in 1.85GB of VRAM.
- **CPU fp32 has the widest p95/p50 spread of anything measured** (79.58 vs 43.42). CPU inference is the most sensitive to output-length variation because there's no parallel headroom to absorb it.

Reading this table:

- **MPS is 5.3× faster than CPU on the same 2.25B weights** (26.9s vs 144.0s mean; 6.62 vs 0.90 tok/s). That gap is the whole argument for the fp16-on-MPS path over a CPU fallback.
- **The p95/p50 spread is large and real** — 39.5s vs 27.4s on MPS. It tracks output length, not backend jitter: the chart image produces several times more tokens than the key-value forms. This is exactly why tokens/sec is in the table.
- **The 256M model is 4.1× faster than the 2.25B on MPS** (6.5s vs 26.9s) and turns the CPU tier from unusable into tolerable (10.3s vs 144.0s). Whether that speed is worth its accuracy cost is the next table's job.
- **RSS and device memory measure different things and neither alone is the answer.** The MPS rows show tiny RSS (335MB) because the weights live in device memory; the CPU rows show no device memory at all and carry everything in RSS. A deployment decision needs both columns.
- **`torch.mps.driver_allocated_memory()` counts the driver's total allocation including its cache**, so the MPS device column overstates the working set — 5,668MB for a 256M-parameter model is cache, not weights. It is an upper bound; RSS is a lower bound; the truth is between them.

### Accuracy vs. quantization level

Speed numbers alone don't answer the question that decides a deployment: *what does the speedup cost you?* `vf accuracy` runs a labeled extraction set through each configuration and reports schema validity and field accuracy alongside latency.

Ground truth is the literal text drawn into the synthetic sample images by `examples/generate_sample_images.py`, so labels are exact by construction rather than human-annotated. **This set is small — 3 images, 21 fields.** It is sized to expose *relative* differences between configurations, not to establish an absolute accuracy claim; a DocVQA/ChartQA subset is the obvious next step and is not done yet.

Two accuracy columns, because either alone misleads: a 2B VLM reading a scanned document routinely drops one character from a long identifier (`SYN-0041729` for `SYN-00417293`). Exact match scores that a total failure; fuzzy match (character similarity ≥ 0.85) scores it a near-miss. The truth is in between.

Every rate is reported with a 95% Wilson confidence interval, and `vf accuracy` prints an explicit "not statistically distinguishable" line naming the configurations whose intervals overlap the leader's. That line exists because ranking configurations off point estimates is the default way to misread a table like this — and because an earlier draft of this README did exactly that.

```bash
vf accuracy
```

For statistical power, build a larger set from a public VQA dataset. `vf dataset` materializes DocVQA (scanned business documents) or ChartQA (charts) into the same labeled-set format, using `{"any_of": [...]}` ground truth so the several acceptable answer spellings those datasets annotate all count as correct:

```bash
vf dataset docvqa -n 200 && vf accuracy --labeled-set benchmarks/datasets/docvqa_200/labeled_set.json
```

A full sweep at that size runs to hours, mostly in the CPU rows — `--only cuda`, `--skip cpu`, and `--limit N` make it tractable. The trade-off is worth knowing: those sets are one-question-one-answer VQA items, so each sample contributes a single field. They give statistical power but exercise the multi-field schema path far less than `labeled_set.json`, where field ordering and partial-object failures are visible. Run both.

#### DocVQA, n=150 — where constrained decoding stops being a nicety

SmolVLM-256M on Apple M3 (MPS, fp16), 150 real scanned-document questions, `max_new_tokens=128`:

| Configuration | Quant | Schema-valid [95% CI] | Field acc. exact [95% CI] | Field acc. fuzzy [95% CI] | Repairs | Tokens/sample (1st + repair) | Mean s/sample |
|---|---|---|---|---|---|---|---|
| Apple MPS / fp16 | fp16-mps | 2% (3/150) [1–6] | 1% (1/150) [0–4] | 1% (1/150) [0–4] | 108 | 74 (9 + 65) | 9.3 |
| Apple MPS / fp16 + constrained | fp16-mps | **79%** (118/150) [71–84] | **39%** (58/150) [31–47] | **43%** (65/150) [36–51] | 17 | 48 (35 + 13) | **7.2** |

**This one is not close, and unlike the 21-field tables it is statistically decisive.** Schema validity [1–6] vs [71–84] and exact accuracy [0–4] vs [31–47] — intervals nowhere near each other at n=150. Unconstrained, this model emits usable JSON on 3 of 150 real documents. Constrained, on 118. Constrained decoding is the difference between a pipeline that works and one that doesn't.

**And the token column falsifies the explanation I gave earlier in this README.** I had written that constrained decoding is faster partly because "the grammar stops generation when the object closes instead of rambling toward the token limit." The data says the reverse: unconstrained first passes average **9 tokens** — not rambling, barely answering — and then spend **65 more** tokens on repair, triggered 108 times out of 150. Constrained first passes are *longer* (35 tokens) because they actually emit a JSON object, and then need only 13 repair tokens.

So the speedup decomposes cleanly, and it has one cause: **avoided repair passes, not shorter generation.** Constrained decoding does more work per first pass and still finishes faster (7.2s vs 9.3s), because it almost never pays for a second forward pass. That is the kind of claim latency alone cannot support and token accounting settles.

Scope this honestly: it is the 256M model on one dataset. A 2.25B model follows instructions better and would likely produce a smaller gap — the synthetic-set tables above show it needing only 1–2 repairs, not 108. What generalizes is the mechanism, not the magnitude.

| Configuration | Quant | Schema-valid | Field acc. (exact) | Field acc. (fuzzy) | Repairs used | Mean s/sample |
|---|---|---|---|---|---|---|
| CUDA / INT4 (bnb nf4) | — | *not measured — no CUDA device on this machine* | — | — | — | — |
| CUDA / INT8 (bnb) | — | *not measured — no CUDA device on this machine* | — | — | — | — |
| Apple MPS / fp16 | fp16-mps | 33% (1/3) | 5% (1/21) | 29% (6/21) | 2 | 38.5 |
| CPU / fp32 | fp32-cpu | 67% (2/3) | **76%** (16/21) | 76% (16/21) | 2 | 144.0 |
| Apple MPS / fp16 (SmolVLM-256M) | fp16-mps | 67% (2/3) | 43% (9/21) | 48% (10/21) | 1 | 4.2 |
| Apple MPS / fp16 + constrained | fp16-mps | 67% (2/3) | 29% (6/21) | **90%** (19/21) | **0** | 28.8 |
| CPU / fp32 + constrained | fp32-cpu | 67% (2/3) | 71% (15/21) | **90%** (19/21) | **0** | **48.0** |
| MPS / fp16 (SmolVLM-256M) + constrained | fp16-mps | **100%** (3/3) | 43% (9/21) | 48% (10/21) | **0** | 5.5 |

Raw per-field outcomes: [`benchmarks/results/accuracy_report.json`](benchmarks/results/accuracy_report.json).

#### NVIDIA L40S (USC CARC)

| Configuration | Quant | Schema-valid [95% CI] | Field acc. exact [95% CI] | Field acc. fuzzy [95% CI] | Repairs | Mean s/sample |
|---|---|---|---|---|---|---|
| CUDA / INT4 (bnb nf4) | int4-bnb-nf4 | 67% (2/3) [21–94] | 38% (8/21) [21–59] | 57% (12/21) [36–75] | 2 | 4.3 |
| CUDA / INT8 (bnb) | int8-bnb | 33% (1/3) [6–79] | 29% (6/21) [14–50] | 29% (6/21) [14–50] | 2 | 11.5 |
| CPU / fp32 | fp32-cpu | 67% (2/3) [21–94] | 76% (16/21) [55–89] | 76% (16/21) [55–89] | 2 | 92.0 |
| CUDA / INT4 + constrained | int4-bnb-nf4 | 100% (3/3) [44–100] | 43% (9/21) [24–63] | 81% (17/21) [60–92] | **0** | **3.4** |
| CUDA / INT8 + constrained | int8-bnb | 67% (2/3) [21–94] | 48% (10/21) [28–68] | 86% (18/21) [65–95] | **0** | 6.8 |
| CPU / fp32 + constrained | fp32-cpu | 67% (2/3) [21–94] | 71% (15/21) [50–86] | 90% (19/21) [70–97] | **0** | 50.9 |

**Read the intervals before the point estimates.** At 21 fields, one field is 4.8 percentage points and most of the gaps in this table are one or two fields. An earlier draft of this README ranked these configurations off point estimates and drew three conclusions that its own data does not support. Corrected:

**What the data does support:**

1. **Constrained decoding eliminates the repair pass.** Zero repairs in all six constrained runs across two architectures, against 1–2 in every unconstrained run. This is a deterministic count, not a proportion estimate, and it reproduces on Apple Silicon and NVIDIA.
2. **Constrained decoding is faster, not slower.** INT4: 4.3s → 3.4s. CPU fp32: 92.0s → 50.9s. MPS fp16: 38.5s → 28.8s. Consistent direction, large margins, every backend. The "correctness enforcement costs latency" intuition is simply wrong on this workload. Token accounting on the n=150 DocVQA run (below) shows why: the saving is entirely avoided repair passes, and constrained first passes are actually *longer*.
3. **fp16 on MPS is genuinely broken for extraction.** 5% exact [1–23] versus fp32's 76% [55–89] — the only accuracy comparison here whose intervals are nowhere near overlapping. And 4-bit NF4 on CUDA scores 38% [21–59], comfortably above 16-bit float on MPS. If precision alone explained it, 16 bits would beat 4 bits; it loses by 7×. Hardware and quantization both differ, so this isn't proof of a specific mechanism — but "fp16 is lossy" is the wrong explanation to settle on, and the MPS path should not be trusted for extraction until a per-layer activation comparison against CUDA explains it.

**What the data does *not* support, despite looking like it does:**

4. **INT8 vs INT4 on accuracy.** 29% [14–50] vs 38% [21–59] — a two-field difference with almost entirely overlapping intervals. The *latency* difference is solid (11.5s vs 4.3s, and 3.7× in the table above, with tight per-run variance), so "prefer INT4" remains the right call on speed and memory alone. But the accuracy claim needs a bigger set.
5. **"Quantization costs accuracy" at the magnitude it appears to.** fp32 76% [55–89] vs INT4 38% [21–59] overlap at the margin. The direction is plausible and matches the fp16 finding, but at this sample size it falls short of significance. What is safe to say: *nothing here demonstrates that quantization is free*, which is the claim most VLM-quantization projects make without a baseline at all.
6. **Any schema-validity comparison.** n=3 documents. "100%" carries a confidence interval of [44–100]. That column is a smoke test, not a measurement.

**The deployment read:** INT4 + constrained is the throughput default — fastest measured anywhere here at 3.4s/sample, with zero repair passes and 1.85GB VRAM. Use fp32 when field-level correctness matters more than latency; the evidence for that preference is directional rather than conclusive, but it points one way consistently. Do not use fp16 on MPS for extraction.

**The headline result is uncomfortable, so it goes first: the fast path is the inaccurate one.** On identical weights and prompts, fp32 on CPU scores 76% exact field accuracy against fp16 on MPS's 5% — while being 3.7× slower. That is not a rounding difference, it's a different quality tier. The likely cause is fp16 range loss in the vision tower degrading the image features, which is a known hazard for fp16 inference on some architectures; this benchmark measures it but does not prove the mechanism. Anyone deploying the MPS path should treat this as the finding that matters most, and it is exactly the kind of result a speed-only benchmark would never surface.

**Constrained decoding eliminated the repair pass entirely** — 0 repairs in all three constrained rows, against 1–2 in every unconstrained one — and raised fuzzy field accuracy to 90% on both 2.25B configurations.

**Constrained decoding was also faster, substantially.** CPU/fp32 went from 144.0s to 48.0s per sample, a 3× speedup; MPS from 38.5s to 28.8s. "Correctness enforcement costs latency" is the intuition here, and on this workload it is simply wrong. The n=150 DocVQA run below decomposes the cause with token counts: it is the avoided repair passes, not shorter generation.

**The 256M model at 100% schema validity in 5.5s** is the deployable configuration for a Pi-class device — provided 43% exact field accuracy is acceptable for the use case, which for triage or routing it may well be and for clinical extraction it is not.

Caveats that limit how far these numbers travel: **21 fields, one seed, three images.** The exact/fuzzy gap on the constrained MPS row (29% vs 90%) means most of its answers are near-misses rather than hits, and fuzzy matching at 0.85 similarity is generous on short strings — `2026-06-02` vs `202-06-02` passes fuzzy and fails exact, which is the intended behavior but should be read as "almost right", not "right". These numbers are sound for ranking configurations against each other and are not a basis for an absolute accuracy claim.

## Constrained decoding

The repair loop in `extractors.py` fixes malformed JSON *after* generation — a second full forward pass, and it can still fail. `constrained.py` moves the guarantee into the decoder: at each step, any token that would make the output impossible to complete as valid JSON has its logit masked to `-inf`. Malformed JSON becomes unrepresentable rather than recoverable.

Three paths, in preference order:

1. **`outlines`**, if installed (`pip install visionflow[constrained]`) — full-vocabulary FSM-guided decoding.
2. **The built-in `JSONPrefixLogitsProcessor`** — no extra dependency, works with any HuggingFace tokenizer, backed by an incremental JSON prefix validator with 39 unit tests covering the prefix/complete/invalid boundaries (including the invariant that *every* prefix of a valid document is accepted, since a false rejection would silently truncate the model mid-object).
3. **GBNF grammars** (`json_schema_to_gbnf`) for the llama.cpp GGUF path, where enforcement happens in C++. With a schema, the grammar pins the exact key sequence, so the model cannot invent, drop, or reorder fields.

**Honest limitation of path 2**: validating all ~49k vocabulary entries per step is too slow in Python, so the built-in processor validates only the top-`k` candidates by logit (default 256) and masks the rest. It still guarantees valid JSON, but it restricts sampling to that window. Under greedy/low-temperature decoding — the default for extraction — the constrained argmax equals the unconstrained argmax whenever the latter is grammar-valid, so in practice this only changes output where the raw model would have emitted invalid JSON. Paths 1 and 3 have no such approximation.

This guarantees **syntactic** validity only. Whether the values are *correct*, and whether they satisfy the schema's semantics, is still the caller's problem — so `constrained=True` keeps the repair pass as a fallback for schema-level failures.

### What it actually changes

Same image, same schema, same greedy decode, SmolVLM-256M on MPS:

| | Raw model output | Parsed result | Repair pass needed | Time |
|---|---|---|---|---|
| Repair loop (v1) | `Synthetic Quarterly Revenue ($M) - Sample Dashboard` — prose, not JSON | `{"title": …, "subtitle": …, "x-axis": …, "y-axis": …}` | **yes** | 3.9s |
| Constrained | `{ "title": …, "q1_revenue": …, "q2_revenue": …, "q3_revenue": … }` | same, parsed directly | no | 3.5s |

Three things to notice. The unconstrained model didn't emit *slightly malformed* JSON — it emitted no JSON at all, and only the repair pass rescued it. That repair invented `subtitle`, `x-axis`, and `y-axis`, none of which are in the requested schema, because it was re-prompted on prose rather than on a partial object. And constraining was **faster**, because skipping a second full forward pass more than pays for the per-step masking (4,237 tokens masked across 68 steps, zero fallbacks).

Both runs still get the *values* wrong — a 256M model misreads the bar values. Constrained decoding buys you syntax and schema conformance, not accuracy. The next table is where accuracy gets measured.

Reproduce: `python examples/constrained_extraction.py`

## ONNX Runtime and TensorRT

`visionflow/onnx_export.py` exports SmolVLM's vision tower to ONNX and benchmarks it across execution providers. Verification status, stated plainly:

| Path | Status |
|---|---|
| ONNX export of the vision encoder | ✅ Verified on Apple M3 — max abs. difference vs PyTorch = **5.04e-04** |
| `CPUExecutionProvider` | ✅ Measured on Apple M3 |
| `CUDAExecutionProvider` | ⚠️ **Available but not yet measured** — present in ORT 1.29 on the CARC L40S; the run's export stage failed on a missing `onnxscript` |
| `TensorrtExecutionProvider` | ⚠️ **Available but not yet measured** — same; ORT registered the provider, so this is pending a re-run, not missing hardware |
| `CoreMLExecutionProvider` | Available but excluded by default — see below |
| Full-model export (decoder + KV cache) | Delegated to `optimum.exporters.onnx`; not exercised here |

```
ONNX export: ok — torchscript+static-position-ids, opset 17, 22.3s,
             input [1, 3, 384, 384] → output [1, 729, 1152],
             max |ONNX − PyTorch| = 5.04e-04
```

| Execution provider | Status | p50 (ms) | p95 (ms) | mean (ms) |
|---|---|---|---|---|
| TensorrtExecutionProvider | *not measured — not available in this onnxruntime build* | — | — | — |
| CUDAExecutionProvider | *not measured — not available in this onnxruntime build* | — | — | — |
| CPUExecutionProvider | ok | 2585.44 | 2675.20 | 2556.77 |

One image patch (384×384) through the vision tower, 20 runs, 3 warmup. Note the tolerance: **5.04e-04 is loose for fp32**, and it is the honest figure — accumulated difference across a deep transformer, not a bit-exact match. It is small enough that downstream token predictions are unaffected, but it is not zero and the table says so.

**Getting this to export at all took a real fix, worth naming.** SmolVLM's vision embeddings compute position ids by counting unmasked patches, bucketizing fractional coordinates, and scattering through a boolean mask. Both ONNX exporters trace that into a `GatherND` whose indices are valid only for the traced batch — the graph exports without complaint, loads without complaint, and then throws `invalid index found, index = 26` on the *very input it was exported with*. For a full, unpadded patch grid that whole computation reduces to `arange(num_patches)`, so the exporter specializes it away. The exported graph is therefore correct for full patch grids and wrong for padded ones — and rather than assume the substitution is safe, the harness compares the exported graph against the **unpatched** PyTorch module, which is where the 5.04e-04 comes from.

CoreML is excluded from the default provider list: it compiles at session-creation time and supports only 768 of this graph's 1607 nodes, so it spends minutes partitioning and then runs a hybrid CoreML/CPU graph whose latency answers a different question. Opt in with `--providers CoreMLExecutionProvider`.

### GPU run on CARC (or any Slurm cluster)

[`scripts/carc_gpu_benchmark.slurm`](scripts/carc_gpu_benchmark.slurm) runs all three harnesses on one GPU node. One-time setup on the **login** node — model weights must be cached there, since compute nodes may have no outbound internet:

```bash
export HF_HOME=/scratch1/$USER/hf     # /home1 quota is too small for a 4.5GB checkpoint
git clone https://github.com/vineetha00/visionflow ~/visionflow && cd ~/visionflow
module purge && module load conda cuda
python -m venv .venv && source .venv/bin/activate
pip install -e ".[cuda,onnx-gpu,eval]"
python -c "from huggingface_hub import snapshot_download as d; [d(m) for m in ['HuggingFaceTB/SmolVLM-Instruct','HuggingFaceTB/SmolVLM-256M-Instruct']]"
```

Then submit, passing your account (find it with `myaccount`). Create `logs/` first — Slurm opens the job's output file before the script body runs, so the script cannot create its own log directory:

```bash
mkdir -p logs && sbatch --account=<your_account> scripts/carc_gpu_benchmark.slurm
```

CUDA is hidden behind a compiler in CARC's Lmod hierarchy, so `module load cuda` alone fails. On Discovery the working pairing is `gcc/12.3.0 cuda/12.4.1`; find yours with `module spider cuda/<version> 2>&1 | cat`. Install torch from the **cu126** index rather than letting pip resolve it — the default wheel there is `+cu130`, and `onnxruntime-gpu` (CUDA 12 + cuDNN 9) and `bitsandbytes` will not work against CUDA 13.

Results land in `benchmarks/results/*_cuda.{json,md}`, written under separate filenames so a GPU run never overwrites the Apple Silicon numbers above.

The TensorRT path is written against the ORT provider API, not from measured experience, and it is labelled that way here rather than presented as a result. ORT falls back silently — requesting `TensorrtExecutionProvider` on a machine without it yields a working CPU session and no error — so the harness records the provider the live session *actually* used and flags any fallback, which means a CPU number can't be mistaken for a TensorRT one. To earn those rows, run this on a CUDA machine:

```bash
pip install "visionflow[onnx-gpu]" && python -m visionflow.onnx_export --verify --benchmark
```

Only the vision encoder is exported by default: it's a single fixed-shape forward pass that exports cleanly, and it's the component most sensitive to execution provider. The autoregressive decoder needs separate prefill/decode graphs with KV-cache tensors as graph I/O — a considerably larger job, left to `optimum`.

## What makes this different from just running SmolVLM

1. **A benchmarking methodology, not a benchmark run** — process isolation, excluded warmup, p50/p95, tokens/sec, and explicit skip rows. The methodology section above documents *why* each choice was made, including the memory-measurement bug it was written to fix.
2. **Correctness moved from prompting into the inference stack** — grammar-constrained decoding makes invalid JSON unrepresentable, with the repair loop demoted to a fallback for schema semantics. On 150 real DocVQA documents this is the difference between 2% and 79% usable output, and it is *faster*.
3. **Capability tiers per device** — model size is part of hardware auto-detection, so the same code path serves an M3 and a Pi without the user choosing a checkpoint.
4. **Accuracy reported against quantization level, not in isolation** — schema validity and field accuracy per configuration, so the speed/accuracy trade-off is visible instead of asserted. This is what surfaced the fp16 accuracy cliff above, which a latency-only benchmark would have reported as a straight 3.7× win.
5. **Honest about what wasn't run** — the CUDA, TensorRT, and GGUF paths are implemented and unmeasured, and they say so in every table rather than being quietly omitted.

## Repo structure

```
visionflow/
├── visionflow/
│   ├── engine.py         # Model loading, hardware detection, capability tiers
│   ├── pipeline.py       # VisionFlow main class — image + prompt → output
│   ├── extractors.py     # Structured extraction: JSON, key-value, repair pass
│   ├── constrained.py    # Grammar-constrained JSON decoding (+ GBNF, outlines)
│   ├── bench.py          # `vf bench` — latency / throughput / memory harness
│   ├── accuracy.py       # `vf accuracy` — accuracy per quantization level
│   ├── onnx_export.py    # ONNX export + execution-provider benchmarking
│   ├── eval.py           # GPT-4o Vision comparison harness
│   ├── quantize.py       # INT4/INT8/GGUF quantization utilities
│   └── cli.py            # `visionflow` / `vf` command-line entry point
├── examples/             # Three use-case scripts + synthetic sample images
├── benchmarks/
│   ├── labeled_set.json  # Ground-truth labels for the accuracy harness
│   └── results/          # Raw benchmark output (JSON + markdown)
├── tests/
├── LICENSE               # MIT
└── CITATION.cff
```

## Known gaps

Listed here rather than discovered later:

- **The fp16-on-MPS accuracy drop is measured but not explained**, and the L40S run makes it stranger rather than clearer: 4-bit NF4 on CUDA beats 16-bit float on MPS by 7× on exact accuracy. That rules out "fp16 is simply lossy" as a sufficient explanation and points at MPS backend numerics, but two variables move at once and it is not proven. Needs a per-layer activation comparison between the MPS and CUDA vision towers. Not fixed here.
- **The headline tables above are still computed on the 21-field hand-built set, and most of their comparisons don't clear their confidence intervals.** Only fp32-vs-fp16-on-MPS does. `vf dataset` now builds DocVQA/ChartQA sets at arbitrary n and the harness reports Wilson intervals, so the tooling gap is closed — but the large-n sweep that would upgrade those directional claims into measurements has only been run on the 256M model so far. The 2.25B INT4/INT8/fp32 comparison at n≥150 is the outstanding work, and [`scripts/carc_accuracy_large.slurm`](scripts/carc_accuracy_large.slurm) runs it.
- **No TensorRT or CUDA execution-provider numbers yet.** The L40S run's ONNX stage failed on a missing `onnxscript` (now added to the `[onnx]` extras), so the export never happened and all three provider rows recorded "graph not found". The providers themselves *are* present in that environment — ORT 1.29 registered `TensorrtExecutionProvider` — so this is a packaging fix away, not a hardware limitation.
- **The 500M tier and the 256M-on-CUDA combination remain unbenchmarked.**
- **The labeled set is 3 images.** Enough to compare configurations, not enough to state an absolute accuracy figure.
- **No GPT-4o Vision baseline was run.** `eval.py` implements the comparison; it requires `OPENAI_API_KEY` and reports `ran: false` with a reason when the key is absent, rather than fabricating a score.
- **GGUF/llama.cpp path unbenchmarked.** Implemented in `quantize.py` including GBNF grammar wiring; no numbers.
- **Constrained decoding's top-k window** is an approximation on the built-in path — see above.
- **The exported ONNX graph covers the vision encoder only, for full patch grids only.** The decoder isn't exported, and the position-id specialization that makes export possible is invalid for padded patches.
- **The 500M (MEDIUM) tier is selected but never benchmarked here.** This machine has 16GB and picks the 2.25B tier; the 256M tier was benchmarked by pinning it explicitly. The 500M path is the same code, but no numbers were measured for it.

## License

MIT — see [LICENSE](LICENSE).

## Citation

See [CITATION.cff](CITATION.cff).
