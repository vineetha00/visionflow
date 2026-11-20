| Configuration | Quant | Schema-valid | Field acc. exact [95% CI] | Field acc. fuzzy [95% CI] | Repairs | Tokens/sample (1st + repair) | Mean s/sample |
|---|---|---|---|---|---|---|---|
| CUDA / INT4 (bnb nf4) | int4-bnb-nf4 | 0% (0/150) [0–2] | 0% (0/150) [0–2] | 0% (0/150) [0–2] | 150 | 104 (7 + 97) | 2.4 |
| CUDA / INT8 (bnb) | int8-bnb | 0% (0/150) [0–2] | 0% (0/150) [0–2] | 0% (0/150) [0–2] | 150 | 104 (7 + 97) | 7.3 |
| CUDA / INT4 (bnb nf4) + constrained | int4-bnb-nf4 | 93% (140/150) [88–96] | 58% (87/150) [50–66] | 69% (103/150) [61–76] | 10 | 32 (24 + 7) | 0.9 |
| CUDA / INT8 (bnb) + constrained | int8-bnb | 93% (139/150) [87–96] | 65% (98/150) [57–72] | 75% (113/150) [68–82] | 11 | 30 (21 + 9) | 2.5 |
| CUDA / INT4 (bnb nf4) + speculative | int4-bnb-nf4 | 100% (150/150) [98–100] | 59% (89/150) [51–67] | 69% (103/150) [61–76] | 3 | 14 (13 + 1) | 0.5 |
| CUDA / INT8 (bnb) + speculative | int8-bnb | 100% (150/150) [98–100] | 67% (101/150) [59–74] | 76% (114/150) [69–82] | 0 | 13 (13 + 0) | 1.1 |
| CUDA / INT4 (bnb nf4) + constrained + speculative | int4-bnb-nf4 | 100% (150/150) [98–100] | 59% (89/150) [51–67] | 69% (103/150) [61–76] | 3 | 14 (13 + 1) | 0.5 |
| CUDA / INT8 (bnb) + constrained + speculative | int8-bnb | 100% (150/150) [98–100] | 67% (101/150) [59–74] | 76% (114/150) [69–82] | 0 | 13 (13 + 0) | 1.1 |

150 images / 150 labeled fields. Exact = normalized string equality; fuzzy = character similarity ≥ 0.85. "Schema-valid" requires parseable JSON containing every requested key. Each configuration measured in an isolated subprocess. Bracketed ranges are 95% Wilson intervals.

**Not statistically distinguishable.** The top configuration by point estimate is *CUDA / INT8 (bnb) + speculative* (67%), but its confidence interval overlaps those of: CUDA / INT4 (bnb nf4) + constrained, CUDA / INT8 (bnb) + constrained, CUDA / INT4 (bnb nf4) + speculative, CUDA / INT4 (bnb nf4) + constrained + speculative, CUDA / INT8 (bnb) + constrained + speculative. With this sample size those differences are not evidence of a real gap.
