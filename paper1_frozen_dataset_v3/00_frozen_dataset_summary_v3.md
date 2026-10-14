# Paper 1 Frozen Dataset Summary — v3

## Final modeling sets

- Development/reference: **2027** unique images
- Reference quarantine: **49** unique images
- Historical external: **1044** unique images
- Field external: **1187** unique images

## Corrected five-fold split

| Fold | N | Multi | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 | 12 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 406 | 9 | 35 | 16 | 30 | 46 | 34 | 45 | 43 | 23 | 28 | 37 | 37 | 41 |
| 1 | 405 | 10 | 35 | 16 | 30 | 46 | 35 | 45 | 43 | 22 | 28 | 37 | 37 | 41 |
| 2 | 405 | 10 | 34 | 16 | 30 | 46 | 35 | 45 | 42 | 22 | 29 | 37 | 37 | 42 |
| 3 | 406 | 9 | 35 | 16 | 30 | 46 | 35 | 46 | 43 | 23 | 28 | 36 | 36 | 42 |
| 4 | 405 | 9 | 34 | 16 | 31 | 45 | 35 | 45 | 42 | 23 | 29 | 37 | 36 | 41 |

## QC

- Cross-domain exact pixel duplicates: **0**.
- Reference records explicitly marked `非目标` or `硬负样本` are quarantined and do not enter supervised model development.
- Historical and field domains remain untouched external evaluation sets.