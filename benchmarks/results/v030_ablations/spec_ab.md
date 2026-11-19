| Configuration | Quant | Schema-valid | Field acc. exact [95% CI] | Field acc. fuzzy [95% CI] | Repairs | Tokens/sample (1st + repair) | Mean s/sample |
|---|---|---|---|---|---|---|---|
| Apple MPS / fp16 | fp16-mps | 0% (0/30) [0–11] | 0% (0/30) [0–11] | 0% (0/30) [0–11] | 30 | 79 (11 + 68) | 22.4 |
| Apple MPS / fp16 + speculative | fp16-mps | 100% (30/30) [89–100] | 33% (10/30) [19–51] | 43% (13/30) [27–61] | 1 | 18 (15 + 3) | 6.8 |

30 images / 30 labeled fields. Exact = normalized string equality; fuzzy = character similarity ≥ 0.85. "Schema-valid" requires parseable JSON containing every requested key. Each configuration measured in an isolated subprocess. Bracketed ranges are 95% Wilson intervals.
