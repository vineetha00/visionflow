| Configuration | Quant | Schema-valid | Field acc. exact [95% CI] | Field acc. fuzzy [95% CI] | Repairs | Tokens/sample (1st + repair) | Mean s/sample |
|---|---|---|---|---|---|---|---|
| Apple MPS / fp16 + constrained | fp16-mps | 87% (26/30) [70–95] | 30% (9/30) [17–48] | 40% (12/30) [25–58] | 4 | 40 (27 + 13) | 8.9 |

30 images / 30 labeled fields. Exact = normalized string equality; fuzzy = character similarity ≥ 0.85. "Schema-valid" requires parseable JSON containing every requested key. Each configuration measured in an isolated subprocess. Bracketed ranges are 95% Wilson intervals.
