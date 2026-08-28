# VisionFlow

Medical images can't go to the cloud. Surgical reports contain PHI. Supply chain manifests are proprietary. Every enterprise team with vision-language needs is stuck choosing between privacy and capability. VisionFlow closes that gap — a fully local, quantized Vision-Language Model pipeline that runs on a MacBook and returns structured data, not a cloud API bill.

VisionFlow loads a small VLM ([SmolVLM](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct)), auto-detects your hardware *and picks a model size that fits it*, runs inference entirely on-device, and constrains the decoder so the output is valid JSON by construction rather than by hope.

**No image, prompt, or extracted field ever leaves the machine.**

## Install

```bash
pip install visionflow
```

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
| NVIDIA GPU, ≥6GB VRAM | SmolVLM-2.25B | `bitsandbytes` INT4 (NF4), INT8 fallback | ⚠️ Implemented, **never run** — no NVIDIA hardware available during development |
| NVIDIA GPU, 3–6GB VRAM | SmolVLM-500M | `bitsandbytes` INT4 | ⚠️ Same |
| Apple Silicon, ≥16GB | SmolVLM-2.25B | fp16 on `mps` | ✅ Measured below |
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

```bash
vf accuracy
```

<!-- ACCURACY_TABLE -->

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
| `CPUExecutionProvider` | ✅ Measured |
| `CUDAExecutionProvider` | ⚠️ **Implemented, never run** — not in this `onnxruntime` build; no NVIDIA hardware available |
| `TensorrtExecutionProvider` | ⚠️ **Implemented, never run** — same |
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

The TensorRT path is written against the ORT provider API, not from measured experience, and it is labelled that way here rather than presented as a result. ORT falls back silently — requesting `TensorrtExecutionProvider` on a machine without it yields a working CPU session and no error — so the harness records the provider the live session *actually* used and flags any fallback, which means a CPU number can't be mistaken for a TensorRT one. To earn those rows, run this on a CUDA machine:

```bash
pip install "visionflow[onnx-gpu]" && python -m visionflow.onnx_export --verify --benchmark
```

Only the vision encoder is exported by default: it's a single fixed-shape forward pass that exports cleanly, and it's the component most sensitive to execution provider. The autoregressive decoder needs separate prefill/decode graphs with KV-cache tensors as graph I/O — a considerably larger job, left to `optimum`.

## What makes this different from just running SmolVLM

1. **A benchmarking methodology, not a benchmark run** — process isolation, excluded warmup, p50/p95, tokens/sec, and explicit skip rows. The methodology section above documents *why* each choice was made, including the memory-measurement bug it was written to fix.
2. **Correctness moved from prompting into the inference stack** — grammar-constrained decoding makes invalid JSON unrepresentable, with the repair loop demoted to a fallback for schema semantics.
3. **Capability tiers per device** — model size is part of hardware auto-detection, so the same code path serves an M3 and a Pi without the user choosing a checkpoint.
4. **Accuracy reported against quantization level, not in isolation** — schema validity and field accuracy per configuration, so the speed/accuracy trade-off is visible instead of asserted.
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

- **No NVIDIA measurements.** The INT4/INT8 bitsandbytes paths and both GPU execution providers are implemented and untested on real hardware.
- **The labeled set is 3 images.** Enough to compare configurations, not enough to state an absolute accuracy figure.
- **No GPT-4o Vision baseline was run.** `eval.py` implements the comparison; it requires `OPENAI_API_KEY` and reports `ran: false` with a reason when the key is absent, rather than fabricating a score.
- **GGUF/llama.cpp path unbenchmarked.** Implemented in `quantize.py`; no numbers.
- **Constrained decoding's top-k window** is an approximation on the built-in path — see above.

## License

MIT — see [LICENSE](LICENSE).

## Citation

See [CITATION.cff](CITATION.cff).
