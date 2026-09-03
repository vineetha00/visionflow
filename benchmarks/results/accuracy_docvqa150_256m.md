| Configuration | Quant | Schema-valid | Field acc. exact [95% CI] | Field acc. fuzzy [95% CI] | Repairs | Tokens/sample (1st + repair) | Mean s/sample |
|---|---|---|---|---|---|---|---|
| Apple MPS / fp16 | fp16-mps | 2% (3/150) [1–6] | 1% (1/150) [0–4] | 1% (1/150) [0–4] | 108 | 74 (9 + 65) | 9.3 |
| Apple MPS / fp16 + constrained | fp16-mps | 79% (118/150) [71–84] | 39% (58/150) [31–47] | 43% (65/150) [36–51] | 17 | 48 (35 + 13) | 7.2 |

150 images / 150 labeled fields. Exact = normalized string equality; fuzzy = character similarity ≥ 0.85. "Schema-valid" requires parseable JSON containing every requested key. Each configuration measured in an isolated subprocess. Bracketed ranges are 95% Wilson intervals.
