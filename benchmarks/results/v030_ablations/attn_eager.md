| Configuration | Model | Quant | Attn | Load (s) | p50 (s) | p95 (s) | mean ± std (s) | tok/s | Peak RSS (MB) | Device mem (MB) |
|---|---|---|---|---|---|---|---|---|---|---|
| CUDA / INT4 (bnb nf4) | SmolVLM-256M-Instruct | — | — | — | — | — | *not measured — no CUDA device on this machine* | — | — | — |
| CUDA / INT8 (bnb) | SmolVLM-256M-Instruct | — | — | — | — | — | *not measured — no CUDA device on this machine* | — | — | — |
| Apple MPS / fp16 | SmolVLM-256M-Instruct | fp16-mps | eager | 4.79 | 15.82 | 18.58 | 14.94 ± 3.47 | 6.49 | 441 | 6,329 |
| CPU / fp32 | SmolVLM-256M-Instruct | fp32-cpu | eager | 3.75 | 29.39 | 36.26 | 29.12 ± 5.88 | 3.33 | 2,798 | — |

Host: macOS-15.1-arm64-arm-64bit-Mach-O · arm · 16.0GB RAM · torch 2.13.0. 3 images × 2 repeats, 1 warmup run(s) excluded, max_new_tokens=128, each configuration in an isolated subprocess.
