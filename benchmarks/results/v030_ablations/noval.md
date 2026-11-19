| Configuration | Quant | Schema-valid | Field acc. exact [95% CI] | Field acc. fuzzy [95% CI] | Repairs | Tokens/sample (1st + repair) | Mean s/sample |
|---|---|---|---|---|---|---|---|
| Apple MPS / fp16 | fp16-mps | 0% (0/30) [0–11] | 0% (0/30) [0–11] | 0% (0/30) [0–11] | 23 | 62 (11 + 51) | 13.2 |

30 images / 30 labeled fields. Exact = normalized string equality; fuzzy = character similarity ≥ 0.85. "Schema-valid" requires parseable JSON containing every requested key. Each configuration measured in an isolated subprocess. Bracketed ranges are 95% Wilson intervals.
