| Configuration | Quant | Schema-valid | Field acc. (exact) | Field acc. (fuzzy) | Repairs used | Mean s/sample |
|---|---|---|---|---|---|---|
| CUDA / INT4 (bnb nf4) | — | *not measured — no CUDA device on this machine* | — | — | — | — |
| CUDA / INT8 (bnb) | — | *not measured — no CUDA device on this machine* | — | — | — | — |
| Apple MPS / fp16 | fp16-mps | 33% (1/3) | 5% (1/21) | 29% (6/21) | 2 | 38.5 |
| CPU / fp32 | fp32-cpu | 67% (2/3) | 76% (16/21) | 76% (16/21) | 2 | 144.0 |
| Apple MPS / fp16 (SmolVLM-256M) | fp16-mps | 67% (2/3) | 43% (9/21) | 48% (10/21) | 1 | 4.2 |
| Apple MPS / fp16 + constrained | fp16-mps | 67% (2/3) | 29% (6/21) | 90% (19/21) | 0 | 28.8 |
| CPU / fp32 + constrained | fp32-cpu | 67% (2/3) | 71% (15/21) | 90% (19/21) | 0 | 48.0 |
| Apple MPS / fp16 (SmolVLM-256M) + constrained | fp16-mps | 100% (3/3) | 43% (9/21) | 48% (10/21) | 0 | 5.5 |

3 images / 21 labeled fields. Exact = normalized string equality; fuzzy = character similarity ≥ 0.85. "Schema-valid" requires parseable JSON containing every requested key. Each configuration measured in an isolated subprocess.
