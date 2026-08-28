| Configuration | Model | Quant | Load (s) | p50 (s) | p95 (s) | mean ± std (s) | tok/s | Peak RSS (MB) | Device mem (MB) |
|---|---|---|---|---|---|---|---|---|---|
| CUDA / INT4 (bnb nf4) | SmolVLM-Instruct | — | — | — | — | *not measured — no CUDA device on this machine* | — | — | — |
| CUDA / INT8 (bnb) | SmolVLM-Instruct | — | — | — | — | *not measured — no CUDA device on this machine* | — | — | — |
| Apple MPS / fp16 | SmolVLM-Instruct | fp16-mps | 8.71 | 27.39 | 39.53 | 26.94 ± 8.55 | 6.62 | 335 | 15,316 |
| CPU / fp32 | SmolVLM-Instruct | fp32-cpu | 9.06 | 152.19 | 204.13 | 144.02 ± 48.83 | 0.90 | 7,435 | — |
| CPU / fp32 (SmolVLM-256M) | SmolVLM-256M-Instruct | fp32-cpu | 3.39 | 10.82 | 12.42 | 10.34 ± 1.68 | 14.83 | 2,850 | — |
| Apple MPS / fp16 (SmolVLM-256M) | SmolVLM-256M-Instruct | fp16-mps | 3.65 | 7.21 | 8.77 | 6.53 ± 2.07 | 22.82 | 1,005 | 5,668 |

Host: macOS-15.1-arm64-arm-64bit-Mach-O · arm · 16.0GB RAM · torch 2.13.0. 3 images × 3 repeats, 1 warmup run(s) excluded, max_new_tokens=256, each configuration in an isolated subprocess.
