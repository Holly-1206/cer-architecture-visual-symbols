# Shared Elements, Different Wholes — data and code

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23360307.svg)](https://doi.org/10.5281/zenodo.23360307)

Archived release: [v1.0.0](https://github.com/Holly-1206/cer-architecture-visual-symbols/releases/tag/v1.0.0) · version DOI [`10.5281/zenodo.23360308`](https://doi.org/10.5281/zenodo.23360308)

Code, annotations and derived results for the article *Shared Elements, Different Wholes: Reading the Architecture of the Chinese Eastern Railway across Album, Field and Machine* (submitted to *Architecture and Culture*).

The repository lets readers check how every number in the article and its Supplementary Materials was computed. The images themselves are not included: the historical photographs are reproduced from a copyrighted facsimile, and the reference images remain under the licences of their original hosts. Their identifiers and sources are listed in `provenance/`.

## Contents

| Folder | Content |
|---|---|
| `code/` | All analysis scripts, in pipeline order (see below) |
| `annotations/` | 5,358 LabelMe annotation files (polygon geometry and labels) for the three image domains, with the embedded image data removed |
| `paper1_frozen_dataset_v3/` | Frozen image-level master table and analytical splits |
| `data/` | Visual-symbol vocabulary (84 codes, Supplementary Table S1) |
| `paper1_model_results/convnext_tiny/` | ConvNeXt-Tiny fold metrics, thresholds, validation and external predictions (model weights not included) |
| `paper1_dinov2_results/` | Frozen DINOv2 embeddings (768-d, `.npy`), prototype similarity, centroid alignment and external metrics |
| `paper1_visual_genealogy_results/` | Element profiles, enrichment, similarity matrices, co-occurrence and continuity |
| `paper1_photo_candidates/` | Ranked exemplar candidates and the manual selection used in the figures |
| `dataset_audit_output/` | Inventory and quality-control tables of the raw annotation archive |
| `results/paper1_analysis_checks_results/` | Robustness checks: near-duplicate groups, counting-unit variants, bootstrap intervals, multi-label coding |
| `results/paper1_strict_external_test_results/` | Strict external test and taxonomy stress test (article Table 2; Supplementary Table S11) |
| `results/paper1_annotation_agreement/` | Blind 200-image agreement sample, reviewer sheet and agreement statistics (Supplementary Table S8) |
| `results/supplementary_tables/` | CSV versions of Supplementary Tables S1, S4–S6 |
| `provenance/` | Source of every analysed image: historical album, reference (Openverse identifier), field (author photograph or other) |

## Dataset

| Domain | Images analysed | Source |
|---|---|---|
| Historical | 1,044 | *Zhongdong tielu da huace* [The Grand Album of the Chinese Eastern Railway], 2013 facsimile (Heilongjiang People's Publishing House) |
| Reference | 2,027 | Openly licensed images retrieved through Openverse |
| Field | 1,187 | 1,151 photographs by the authors (April–September 2026); 36 Amap street-view captures and Douyin images |

The master table `paper1_frozen_dataset_v3/01_unique_image_multilabel_master.csv` has 4,370 rows: the 4,258 analysed images (`dataset_role` = `development_reference`, `external_historical`, `external_field`), 49 reference images quarantined as off-target or hard-negative examples (`qc_reference_quarantine`), and 63 images excluded as non-architectural or outside the scope of the style model (`qc_excluded_from_style_model`).

Key columns: `pixel_sha256` (image identifier), `y_01`–`y_12` (lineage labels), `symbol_counts_json` (A–E symbol counts), `fold` (reference cross-validation fold), `dataset_role`. Paths in the tables are relative to the repository root; image paths point to `images/`, which is not distributed.

The twelve lineages: 01 Russian Traditional, 02 Russian Timber Dwelling, 03 Russian Railway, 04 Byzantine/Orthodox, 05 Eclecticism, 06 Art Nouveau, 07 Baroque, 08 Chinese Baroque, 09 Classical/Neoclassical, 10 Gothic, 11 Art Deco, 12 Early Modernism.

## Pipeline

Run from the repository root. Scripts locate files relative to the repository; set the environment variable `CER_PROJECT_ROOT` to use another location.

| Step | Script | Needs images | Output |
|---|---|---|---|
| 1 | `audit_labelme_dataset.py` | yes | `dataset_audit_output/` |
| 2 | `prepare_labelme_corpus_v2.py` | yes | recovered images and master manifest |
| 3 | `freeze_paper1_multilabel_dataset.py` | yes | first frozen master |
| 4 | `repair_paper1_freeze_v3.py` | no | `paper1_frozen_dataset_v3/` |
| 5 | `train_paper1_multilabel.py --model convnext_tiny` | yes | `paper1_model_results/` |
| 6 | `paper1_dinov2_crossdomain.py --model dinov2_vitb14 --image-size 518` | yes | `paper1_dinov2_results/` |
| 7 | `paper1_visual_genealogy_final.py` | no | `paper1_visual_genealogy_results/` |
| 8 | `paper1_analysis_checks.py` | no | `results/paper1_analysis_checks_results/` |
| 9 | `paper1_annotation_agreement.py score` | no | `results/paper1_annotation_agreement/` |
| 10 | `paper1_strict_external_test.py`, `paper1_strict_linear_probe.py` | no | `results/paper1_strict_external_test_results/` |
| — | `paper1_select_photo_candidates_fixed.py`, `paper1_architecture_aware_rerank_03_08.py`, `paper1_figures_3_to_7_v4.py`, `paper1_figure6_two_layer_similarity.py` | yes | figures and exemplar selection |

Steps 4 and 7–10 run from the files in this repository alone and reproduce every statistic reported in the article and the Supplementary Materials. Steps that need images can be rerun after reconstructing the corpus from `provenance/`.

Random seeds: data split 20260916; ConvNeXt training 20260916; DINOv2 and linear probe 20260917; genealogy permutations 20260917; robustness checks 20261010; agreement sample 20261011; strict test 20261014.

## Requirements

Python 3.10 or later; `numpy`, `pandas`, `scipy`, `scikit-learn`, `pillow`, `matplotlib`, `networkx`, `openpyxl`, `tabulate`; `torch` and `torchvision` for steps 5–6.

```
pip install numpy pandas scipy scikit-learn pillow matplotlib networkx openpyxl tabulate torch torchvision
```

## Licence

Code: MIT License (see `LICENSE`). Annotations and derived tables: CC BY 4.0. Images are not included and remain subject to the rights of their holders (see `provenance/`).

## Citation

Lei, H. 2026. "Shared Elements, Different Wholes: Reading the Architecture of the Chinese Eastern Railway across Album, Field and Machine." *Architecture and Culture*. [DOI]. Data and code: https://doi.org/10.5281/zenodo.23360308.
