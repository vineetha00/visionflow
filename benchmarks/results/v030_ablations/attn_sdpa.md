| Configuration | Model | Quant | Attn | Load (s) | p50 (s) | p95 (s) | mean ± std (s) | tok/s | Peak RSS (MB) | Device mem (MB) |
|---|---|---|---|---|---|---|---|---|---|---|
| CUDA / INT4 (bnb nf4) | SmolVLM-256M-Instruct | — | — | — | — | — | *not measured — no CUDA device on this machine* | — | — | — |
| CUDA / INT8 (bnb) | SmolVLM-256M-Instruct | — | — | — | — | — | *not measured — no CUDA device on this machine* | — | — | — |
| Apple MPS / fp16 | SmolVLM-256M-Instruct | fp16-mps | sdpa | 6.43 | 11.57 | 15.82 | 11.45 ± 3.66 | 8.47 | 507 | 4,388 |
| CPU / fp32 | SmolVLM-256M-Instruct | fp32-cpu | sdpa | 3.40 | 16.56 | 20.51 | 16.90 ± 2.80 | 5.74 | 2,217 | — |

Host: macOS-15.1-arm64-arm-64bit-Mach-O · arm · 16.0GB RAM · torch 2.13.0. 3 images × 2 repeats, 1 warmup run(s) excluded, max_new_tokens=128, each configuration in an isolated subprocess.
