# Paper 1 — analysis checks for the advisor review

Data: `paper1_frozen_dataset_v3`  
Seed: 20261010

## Check 2 — Unit of multi-label attribution (C6)

Heuristic unit of multi-label attribution (from per-lineage LabelMe geometry):

| domain     |   likely_single_building |   ambiguous |   image_level_unresolved |   likely_multiple_objects |   total |
|:-----------|-------------------------:|------------:|-------------------------:|--------------------------:|--------:|
| field      |                        1 |           0 |                      261 |                         0 |     262 |
| historical |                      136 |          60 |                        1 |                         1 |     198 |
| reference  |                       43 |           4 |                        0 |                         0 |      47 |

Spatial pattern of lineage annotations:

| domain     |   co_located |   no_geometry_for_some_lineage |   partial_overlap |   shared_annotation |   spatially_separate |
|:-----------|-------------:|-------------------------------:|------------------:|--------------------:|---------------------:|
| field      |            1 |                              0 |                 0 |                 261 |                    0 |
| historical |          181 |                              7 |                 8 |                   1 |                    1 |
| reference  |           46 |                              0 |                 1 |                   0 |                    0 |

Manual coding: 127 images coded.

| heuristic_unit          |   building_plus_context |   multiple_buildings |   single_building |   unclear |
|:------------------------|------------------------:|---------------------:|------------------:|----------:|
| ambiguous               |                       1 |                   25 |                12 |        11 |
| image_level_unresolved  |                      10 |                    5 |                29 |         2 |
| likely_multiple_objects |                       0 |                    1 |                 0 |         0 |
| likely_single_building  |                       0 |                    3 |                22 |         6 |

Stratum-weighted estimate of the multi-label unit:

| domain     |   n_coded |   n_multilabel |   est_prop_single_building |   est_prop_multiple_buildings |   est_prop_building_plus_context |   est_prop_unclear |
|:-----------|----------:|---------------:|---------------------------:|------------------------------:|---------------------------------:|-------------------:|
| reference  |        19 |             47 |                      0.835 |                         0.165 |                            0     |              0     |
| historical |        62 |            198 |                      0.439 |                         0.206 |                            0.007 |              0.349 |
| field      |        46 |            262 |                      0.624 |                         0.111 |                            0.221 |              0.044 |

Observed co-occurrence with likely multi-object images removed:

| domain     |   n_images_all |   n_images_restricted |   spearman_all_vs_restricted |   DINO_vs_cooc_rho_all |   DINO_vs_cooc_p_all |   DINO_vs_cooc_rho_restricted |   DINO_vs_cooc_p_restricted |   AD_vs_cooc_rho_all |   AD_vs_cooc_p_all |   AD_vs_cooc_rho_restricted |   AD_vs_cooc_p_restricted |
|:-----------|---------------:|----------------------:|-----------------------------:|-----------------------:|---------------------:|------------------------------:|----------------------------:|---------------------:|-------------------:|----------------------------:|--------------------------:|
| historical |           1044 |                  1043 |                            1 |                 0.0735 |               0.6842 |                        0.0731 |                      0.6762 |               0.2011 |             0.3075 |                      0.2021 |                    0.3145 |
| field      |           1187 |                  1187 |                            1 |                -0.0128 |               0.9292 |                       -0.0128 |                      0.9353 |               0.1514 |             0.3474 |                      0.1514 |                    0.3477 |
- historical top-5 all: 01-04 0.129; 03-05 0.111; 05-09 0.104; 03-12 0.094; 11-12 0.085
- historical top-5 restricted: 01-04 0.129; 03-05 0.111; 05-09 0.104; 03-12 0.094; 11-12 0.085
- field top-5 all: 02-04 0.259; 07-08 0.203; 01-02 0.180; 01-04 0.159; 01-05 0.135
- field top-5 restricted: 02-04 0.259; 07-08 0.203; 01-02 0.180; 01-04 0.159; 01-05 0.135

## Check 1 — Building/site grouping and leakage (C4, C8)

Visual near-duplicate rule: DINOv2 cosine >= 0.9, or dHash Hamming <= 6 with cosine >= 0.70.
Near-duplicate pairs found: 245 (2 cross-domain).

| domain     |   n_images |   n_name_or_site_keys |   n_groups_after_visual_merge |   n_images_with_visual_neighbour |   n_images_in_cross_domain_groups |
|:-----------|-----------:|----------------------:|------------------------------:|---------------------------------:|----------------------------------:|
| reference  |       2027 |                  2027 |                          1912 |                              199 |                                 2 |
| historical |       1044 |                   907 |                           902 |                               18 |                                 0 |
| field      |       1187 |                   275 |                           258 |                              112 |                                30 |

Reference images whose near-duplicate group spans another CV fold: 165 / 2027 (8.1%).

Five-fold CV, all validation images vs. clean (leaked removed):

| subset   |   ('macro_f1', 'mean') |   ('macro_f1', 'std') |   ('mAP', 'mean') |   ('mAP', 'std') |   ('macro_AUROC', 'mean') |   ('macro_AUROC', 'std') |
|:---------|-----------------------:|----------------------:|------------------:|-----------------:|--------------------------:|-------------------------:|
| all      |                 0.5626 |                0.006  |            0.5375 |           0.0048 |                    0.8459 |                   0.0075 |
| clean    |                 0.5417 |                0.0066 |            0.5129 |           0.0086 |                    0.8374 |                   0.0079 |

External evaluation, all vs. clean (images sharing a group with any reference image removed):

| domain     | subset   |   n_images |   ConvNeXt_macro_f1 |   ConvNeXt_micro_f1 |   ConvNeXt_macro_AUROC |   ConvNeXt_mAP |   DINOv2_prototype_macro_AUROC |   DINOv2_prototype_mAP |
|:-----------|:---------|-----------:|--------------------:|--------------------:|-----------------------:|---------------:|-------------------------------:|-----------------------:|
| historical | all      |       1044 |              0.093  |              0.1838 |                 0.6184 |         0.1741 |                         0.7301 |                 0.3111 |
| historical | clean    |       1044 |              0.093  |              0.1838 |                 0.6184 |         0.1741 |                         0.7301 |                 0.3111 |
| field      | all      |       1187 |              0.145  |              0.2039 |                 0.6167 |         0.1914 |                         0.6627 |                 0.2447 |
| field      | clean    |       1157 |              0.1429 |              0.2008 |                 0.6125 |         0.1874 |                         0.6593 |                 0.2396 |

Field provenance:

| provenance      |   n_images |
|:----------------|-----------:|
| camera_sequence |        363 |
| named_site      |        788 |
| screenshot      |         22 |
| web_image       |         14 |

Group-aware split written to `check1_leakage/paper1_frozen_dataset_v3_grouped/` (67 reference groups were split across folds in v3). Optional re-run:

```
python train_paper1_multilabel.py --model convnext_tiny --data-dir "results/paper1_analysis_checks_results/check1_leakage/paper1_frozen_dataset_v3_grouped" --results-dir "paper1_model_results_grouped"
```

## Check 3 — Annotation density and counting unit (C11, C14)

Sanity: A–D baseline (instance, alpha = 0.5) — 03-05 0.922 (manuscript 0.922); 11-12 0.981 (manuscript 0.981); 01-04 0.814 (manuscript 0.814); 03-10 0.335 (manuscript 0.335). Recomputed DINOv2 prototypes reproduce the published DINOv2 matrix (max |diff| = 3.8e-08).

Note: `05_symbol_similarity_reference.csv` on disk differs from the A–D-only recomputation (max |diff| = 0.020); it appears to be from the earlier A–E run. Re-run paper1_visual_genealogy_final.py before archiving the supplement.

| counting_unit    |   spearman_with_published_AD |   permutation_p_with_published_AD |   spearman_with_DINOv2 |   permutation_p_with_DINOv2 |   AD_median |   region_agreement_with_published |   region_kappa_with_published |   n_High–High |   n_High vocabulary / Low syntax |   n_Low vocabulary / High syntax |   n_Low–Low |
|:-----------------|-----------------------------:|----------------------------------:|-----------------------:|----------------------------:|------------:|----------------------------------:|------------------------------:|--------------:|---------------------------------:|---------------------------------:|------------:|
| instance         |                       1      |                            0.0001 |                 0.334  |                      0.0721 |      0.7531 |                            1      |                        1      |            20 |                               13 |                               13 |          20 |
| presence         |                       0.8773 |                            0.0001 |                 0.3969 |                      0.0083 |      0.6196 |                            0.8788 |                        0.8366 |            19 |                               14 |                               14 |          19 |
| image_normalised |                       0.8471 |                            0.0001 |                 0.5984 |                      0.0001 |      0.673  |                            0.8182 |                        0.7495 |            24 |                                9 |                                9 |          24 |

Key manuscript pairs — region by counting unit:

| pair   | image_normalised             | instance                     | presence                     |
|:-------|:-----------------------------|:-----------------------------|:-----------------------------|
| 01-04  | High–High                    | High–High                    | High–High                    |
| 03-05  | High vocabulary / Low syntax | High vocabulary / Low syntax | High vocabulary / Low syntax |
| 03-10  | Low–Low                      | Low–Low                      | Low–Low                      |
| 04-07  | Low vocabulary / High syntax | Low vocabulary / High syntax | Low vocabulary / High syntax |
| 05-09  | High–High                    | Low vocabulary / High syntax | Low vocabulary / High syntax |
| 06-11  | High–High                    | High–High                    | High–High                    |
| 11-12  | High–High                    | High–High                    | High–High                    |

Key manuscript pairs — A–D similarity (rank of 66) by counting unit:

| pair   |   ('AD_rank', 'image_normalised') |   ('AD_rank', 'instance') |   ('AD_rank', 'presence') |   ('AD_similarity', 'image_normalised') |   ('AD_similarity', 'instance') |   ('AD_similarity', 'presence') |
|:-------|----------------------------------:|--------------------------:|--------------------------:|----------------------------------------:|--------------------------------:|--------------------------------:|
| 01-04  |                                 7 |                        24 |                         5 |                                   0.834 |                           0.814 |                           0.875 |
| 03-05  |                                33 |                         5 |                        14 |                                   0.678 |                           0.922 |                           0.775 |
| 03-10  |                                66 |                        65 |                        63 |                                   0.316 |                           0.335 |                           0.39  |
| 04-07  |                                40 |                        46 |                        51 |                                   0.625 |                           0.598 |                           0.551 |
| 05-09  |                                32 |                        37 |                        34 |                                   0.683 |                           0.72  |                           0.616 |
| 06-11  |                                 8 |                         2 |                        10 |                                   0.834 |                           0.938 |                           0.846 |
| 11-12  |                                 1 |                         1 |                         1 |                                   0.975 |                           0.981 |                           0.936 |

Lineage annotation density (canonical single-lineage reference images):

|   style_id | style_name_en            |   n_canonical_images |   mean_AD_instances_per_image |   mean_AD_types_per_image |   share_images_without_AD |   mean_offdiag_similarity_instance |   mean_offdiag_similarity_presence |   mean_offdiag_similarity_image_normalised |
|-----------:|:-------------------------|---------------------:|------------------------------:|--------------------------:|--------------------------:|-----------------------------------:|-----------------------------------:|-------------------------------------------:|
|         01 | Russian Traditional      |                  135 |                        17.83  |                     6.178 |                     0     |                              0.627 |                              0.579 |                                      0.62  |
|         02 | Russian Timber Dwelling  |                   75 |                        15.92  |                     6.787 |                     0     |                              0.724 |                              0.649 |                                      0.72  |
|         03 | Russian Railway          |                  150 |                         9.807 |                     3.207 |                     0.12  |                              0.76  |                              0.637 |                                      0.563 |
|         04 | Byzantine / Orthodox     |                  210 |                        18.095 |                     5.795 |                     0.014 |                              0.592 |                              0.541 |                                      0.547 |
|         05 | Eclecticism              |                  166 |                        14.53  |                     7.446 |                     0     |                              0.776 |                              0.689 |                                      0.737 |
|         06 | Art Nouveau              |                  223 |                         6.596 |                     2.547 |                     0.018 |                              0.814 |                              0.739 |                                      0.751 |
|         07 | Baroque                  |                  206 |                         8.16  |                     2.665 |                     0     |                              0.749 |                              0.703 |                                      0.717 |
|         08 | Chinese Baroque          |                  112 |                         6.036 |                     2.089 |                     0     |                              0.735 |                              0.689 |                                      0.666 |
|         09 | Classical / Neoclassical |                  138 |                         6.5   |                     2.587 |                     0     |                              0.717 |                              0.688 |                                      0.685 |
|         10 | Gothic                   |                  180 |                         5.483 |                     2.344 |                     0.006 |                              0.429 |                              0.527 |                                      0.481 |
|         11 | Art Deco                 |                  179 |                         7.48  |                     2.603 |                     0.006 |                              0.768 |                              0.659 |                                      0.714 |
|         12 | Early Modernism          |                  206 |                         8.893 |                     2.956 |                     0.005 |                              0.767 |                              0.677 |                                      0.711 |

Spearman (n = 12) between lineage mean A–D instances/image and mean off-diagonal similarity: instance = -0.063, presence = -0.294, image_normalised = 0.028

Annotation density by domain:

| domain     |   n_images |   mean_AD_instances_per_image |   median_AD_instances_per_image |   mean_AD_types_per_image |   share_images_without_AD |
|:-----------|-----------:|------------------------------:|--------------------------------:|--------------------------:|--------------------------:|
| reference  |       2027 |                        10.496 |                               8 |                     3.861 |                     0.014 |
| historical |       1044 |                         7.868 |                               6 |                     3.414 |                     0.096 |
| field      |       1187 |                         5.069 |                               4 |                     2.814 |                     0.05  |

Mean A–D continuity (adequate support only) by counting unit:

| counting_unit    |   field |   historical |
|:-----------------|--------:|-------------:|
| image_normalised |   0.532 |        0.5   |
| instance         |   0.45  |        0.431 |
| presence         |   0.474 |        0.461 |

Spearman of lineage-level continuity with the instance version: instance = 1.000, presence = 0.870, image_normalised = 0.821

Equal-n subsampling (75 images per lineage, 1000 reps):

| counting_unit   | statistic     |   n_per_lineage |   median |   q2.5 |   q97.5 |
|:----------------|:--------------|----------------:|---------:|-------:|--------:|
| instance        | rho_dino      |              75 |    0.336 |  0.243 |   0.432 |
| instance        | rho_published |              75 |    0.964 |  0.92  |   0.985 |
| presence        | rho_dino      |              75 |    0.391 |  0.309 |   0.489 |
| presence        | rho_published |              75 |    0.862 |  0.774 |   0.911 |

Joint bootstrap of DINOv2–A–D rho (2000 reps, images resampled within lineage, both layers recomputed):

| counting_unit   |   observed_rho |   bootstrap_median |   ci95_low |   ci95_high |   share_rho_gt_0 |
|:----------------|---------------:|-------------------:|-----------:|------------:|-----------------:|
| instance        |          0.334 |              0.344 |      0.226 |       0.461 |                1 |
| presence        |          0.397 |              0.401 |      0.29  |       0.51  |                1 |

## Check 4 — Paired bootstrap of external performance (C13)

Sanity (fast metrics vs sklearn on the full sample):
- historical ConvNeXt-Tiny: macro-AUROC 0.6184 (sklearn 0.6184), mAP 0.1741 (sklearn 0.1741)
- historical DINOv2 prototype: macro-AUROC 0.7301 (sklearn 0.7301), mAP 0.3111 (sklearn 0.3111)
- historical DINOv2 linear probe: macro-AUROC 0.6390 (sklearn 0.6390), mAP 0.1597 (sklearn 0.1597)
- field ConvNeXt-Tiny: macro-AUROC 0.6167 (sklearn 0.6167), mAP 0.1914 (sklearn 0.1914)
- field DINOv2 prototype: macro-AUROC 0.6627 (sklearn 0.6627), mAP 0.2447 (sklearn 0.2447)
- field DINOv2 linear probe: macro-AUROC 0.5859 (sklearn 0.5859), mAP 0.1650 (sklearn 0.1650)

95% percentile CIs, 2000 bootstrap replicates. `cluster` resamples building/site groups from CHECK 1 instead of single images.

| domain     | level   | method              | metric      |   point |   ci95_low |   ci95_high |
|:-----------|:--------|:--------------------|:------------|--------:|-----------:|------------:|
| historical | image   | ConvNeXt-Tiny       | macro_AUROC |  0.6184 |     0.5797 |      0.6604 |
| historical | image   | ConvNeXt-Tiny       | mAP         |  0.1741 |     0.1578 |      0.2121 |
| historical | image   | DINOv2 prototype    | macro_AUROC |  0.7301 |     0.6828 |      0.7672 |
| historical | image   | DINOv2 prototype    | mAP         |  0.3111 |     0.2725 |      0.374  |
| historical | image   | DINOv2 linear probe | macro_AUROC |  0.639  |     0.6047 |      0.6753 |
| historical | image   | DINOv2 linear probe | mAP         |  0.1597 |     0.153  |      0.1881 |
| historical | image   | ConvNeXt-Tiny       | macro_F1    |  0.093  |     0.0675 |      0.1218 |
| historical | cluster | ConvNeXt-Tiny       | macro_AUROC |  0.6184 |     0.5774 |      0.6588 |
| historical | cluster | ConvNeXt-Tiny       | mAP         |  0.1741 |     0.1564 |      0.2094 |
| historical | cluster | DINOv2 prototype    | macro_AUROC |  0.7301 |     0.6797 |      0.7681 |
| historical | cluster | DINOv2 prototype    | mAP         |  0.3111 |     0.2706 |      0.374  |
| historical | cluster | DINOv2 linear probe | macro_AUROC |  0.639  |     0.6052 |      0.6735 |
| historical | cluster | DINOv2 linear probe | mAP         |  0.1597 |     0.152  |      0.1887 |
| historical | cluster | ConvNeXt-Tiny       | macro_F1    |  0.093  |     0.0658 |      0.1153 |
| field      | image   | ConvNeXt-Tiny       | macro_AUROC |  0.6167 |     0.592  |      0.6411 |
| field      | image   | ConvNeXt-Tiny       | mAP         |  0.1914 |     0.1769 |      0.2177 |
| field      | image   | DINOv2 prototype    | macro_AUROC |  0.6627 |     0.6404 |      0.6849 |
| field      | image   | DINOv2 prototype    | mAP         |  0.2447 |     0.2277 |      0.2743 |
| field      | image   | DINOv2 linear probe | macro_AUROC |  0.5859 |     0.5589 |      0.6129 |
| field      | image   | DINOv2 linear probe | mAP         |  0.165  |     0.1534 |      0.1887 |
| field      | image   | ConvNeXt-Tiny       | macro_F1    |  0.145  |     0.1198 |      0.1675 |
| field      | cluster | ConvNeXt-Tiny       | macro_AUROC |  0.6167 |     0.539  |      0.6566 |
| field      | cluster | ConvNeXt-Tiny       | mAP         |  0.1914 |     0.1442 |      0.2443 |
| field      | cluster | DINOv2 prototype    | macro_AUROC |  0.6627 |     0.6096 |      0.6881 |
| field      | cluster | DINOv2 prototype    | mAP         |  0.2447 |     0.1861 |      0.3076 |
| field      | cluster | DINOv2 linear probe | macro_AUROC |  0.5859 |     0.5238 |      0.6177 |
| field      | cluster | DINOv2 linear probe | mAP         |  0.165  |     0.1303 |      0.2177 |
| field      | cluster | ConvNeXt-Tiny       | macro_F1    |  0.145  |     0.0835 |      0.1785 |

Paired differences:

| domain     | level   | comparison                             | metric      |   point_difference |   ci95_low |   ci95_high |   share_difference_gt_0 |   bootstrap_p_two_sided |
|:-----------|:--------|:---------------------------------------|:------------|-------------------:|-----------:|------------:|------------------------:|------------------------:|
| historical | image   | DINOv2 prototype − ConvNeXt-Tiny       | macro_AUROC |             0.1117 |     0.0485 |      0.1665 |                  1      |                  0.001  |
| historical | image   | DINOv2 prototype − ConvNeXt-Tiny       | mAP         |             0.137  |     0.0923 |      0.187  |                  1      |                  0.001  |
| historical | image   | DINOv2 prototype − DINOv2 linear probe | macro_AUROC |             0.0911 |     0.0531 |      0.1313 |                  1      |                  0.001  |
| historical | image   | DINOv2 prototype − DINOv2 linear probe | mAP         |             0.1514 |     0.1115 |      0.1996 |                  1      |                  0.001  |
| historical | image   | DINOv2 linear probe − ConvNeXt-Tiny    | macro_AUROC |             0.0206 |    -0.0317 |      0.0799 |                  0.744  |                  0.5127 |
| historical | image   | DINOv2 linear probe − ConvNeXt-Tiny    | mAP         |            -0.0144 |    -0.0414 |      0.0086 |                  0.1195 |                  0.2399 |
| historical | cluster | DINOv2 prototype − ConvNeXt-Tiny       | macro_AUROC |             0.1117 |     0.0408 |      0.1683 |                  1      |                  0.001  |
| historical | cluster | DINOv2 prototype − ConvNeXt-Tiny       | mAP         |             0.137  |     0.0968 |      0.1866 |                  1      |                  0.001  |
| historical | cluster | DINOv2 prototype − DINOv2 linear probe | macro_AUROC |             0.0911 |     0.0528 |      0.1285 |                  1      |                  0.001  |
| historical | cluster | DINOv2 prototype − DINOv2 linear probe | mAP         |             0.1514 |     0.1102 |      0.1983 |                  1      |                  0.001  |
| historical | cluster | DINOv2 linear probe − ConvNeXt-Tiny    | macro_AUROC |             0.0206 |    -0.0365 |      0.0759 |                  0.754  |                  0.4928 |
| historical | cluster | DINOv2 linear probe − ConvNeXt-Tiny    | mAP         |            -0.0144 |    -0.039  |      0.0095 |                  0.1335 |                  0.2679 |
| field      | image   | DINOv2 prototype − ConvNeXt-Tiny       | macro_AUROC |             0.046  |     0.0233 |      0.0714 |                  1      |                  0.001  |
| field      | image   | DINOv2 prototype − ConvNeXt-Tiny       | mAP         |             0.0533 |     0.0314 |      0.0779 |                  1      |                  0.001  |
| field      | image   | DINOv2 prototype − DINOv2 linear probe | macro_AUROC |             0.0768 |     0.0538 |      0.1005 |                  1      |                  0.001  |
| field      | image   | DINOv2 prototype − DINOv2 linear probe | mAP         |             0.0797 |     0.0599 |      0.1013 |                  1      |                  0.001  |
| field      | image   | DINOv2 linear probe − ConvNeXt-Tiny    | macro_AUROC |            -0.0308 |    -0.0572 |     -0.0034 |                  0.0145 |                  0.03   |
| field      | image   | DINOv2 linear probe − ConvNeXt-Tiny    | mAP         |            -0.0264 |    -0.0495 |     -0.0048 |                  0.012  |                  0.025  |
| field      | cluster | DINOv2 prototype − ConvNeXt-Tiny       | macro_AUROC |             0.046  |     0.007  |      0.0974 |                  0.99   |                  0.021  |
| field      | cluster | DINOv2 prototype − ConvNeXt-Tiny       | mAP         |             0.0533 |     0.024  |      0.0843 |                  1      |                  0.001  |
| field      | cluster | DINOv2 prototype − DINOv2 linear probe | macro_AUROC |             0.0768 |     0.0328 |      0.1114 |                  1      |                  0.001  |
| field      | cluster | DINOv2 prototype − DINOv2 linear probe | mAP         |             0.0797 |     0.041  |      0.1159 |                  1      |                  0.001  |
| field      | cluster | DINOv2 linear probe − ConvNeXt-Tiny    | macro_AUROC |            -0.0308 |    -0.0637 |      0.0136 |                  0.0985 |                  0.1979 |
| field      | cluster | DINOv2 linear probe − ConvNeXt-Tiny    | mAP         |            -0.0264 |    -0.0437 |     -0.0061 |                  0.0045 |                  0.01   |

Per-lineage AUROC difference, DINOv2 prototype − ConvNeXt (image-level):

| domain     |   style_id | style_name_en            |   n_positive |   AUROC_DINOv2_prototype |   AUROC_ConvNeXt |   difference |   ci95_low |   ci95_high |   ci_excludes_0 |
|:-----------|-----------:|:-------------------------|-------------:|-------------------------:|-----------------:|-------------:|-----------:|------------:|----------------:|
| historical |         01 | Russian Traditional      |           11 |                    0.985 |            0.728 |        0.258 |      0.074 |       0.48  |               1 |
| historical |         02 | Russian Timber Dwelling  |           42 |                    0.792 |            0.625 |        0.168 |      0.065 |       0.271 |               1 |
| historical |         03 | Russian Railway          |          323 |                    0.733 |            0.587 |        0.146 |      0.105 |       0.189 |               1 |
| historical |         04 | Byzantine / Orthodox     |           68 |                    0.628 |            0.646 |       -0.018 |     -0.133 |       0.097 |               0 |
| historical |         05 | Eclecticism              |          146 |                    0.706 |            0.609 |        0.097 |      0.038 |       0.156 |               1 |
| historical |         06 | Art Nouveau              |           12 |                    0.884 |            0.653 |        0.231 |      0.084 |       0.374 |               1 |
| historical |         07 | Baroque                  |            3 |                    0.712 |            0.357 |        0.355 |     -0.289 |       0.829 |               0 |
| historical |         08 | Chinese Baroque          |          296 |                    0.81  |            0.677 |        0.133 |      0.099 |       0.169 |               1 |
| historical |         09 | Classical / Neoclassical |           24 |                    0.834 |            0.79  |        0.043 |     -0.071 |       0.152 |               0 |
| historical |         10 | Gothic                   |            8 |                    0.861 |            0.696 |        0.166 |     -0.053 |       0.411 |               0 |
| historical |         11 | Art Deco                 |           29 |                    0.437 |            0.477 |       -0.04  |     -0.157 |       0.079 |               0 |
| historical |         12 | Early Modernism          |          304 |                    0.378 |            0.576 |       -0.198 |     -0.248 |      -0.146 |               1 |
| field      |         01 | Russian Traditional      |          147 |                    0.548 |            0.509 |        0.038 |     -0.026 |       0.101 |               0 |
| field      |         02 | Russian Timber Dwelling  |           89 |                    0.554 |            0.514 |        0.039 |     -0.013 |       0.093 |               0 |
| field      |         03 | Russian Railway          |          632 |                    0.562 |            0.63  |       -0.068 |     -0.104 |      -0.032 |               1 |
| field      |         04 | Byzantine / Orthodox     |           86 |                    0.66  |            0.639 |        0.021 |     -0.055 |       0.1   |               0 |
| field      |         05 | Eclecticism              |          172 |                    0.56  |            0.626 |       -0.067 |     -0.109 |      -0.023 |               1 |
| field      |         06 | Art Nouveau              |          183 |                    0.779 |            0.683 |        0.095 |      0.05  |       0.138 |               1 |
| field      |         07 | Baroque                  |           44 |                    0.847 |            0.661 |        0.186 |      0.096 |       0.281 |               1 |
| field      |         08 | Chinese Baroque          |           27 |                    0.878 |            0.628 |        0.25  |      0.149 |       0.361 |               1 |
| field      |         09 | Classical / Neoclassical |           83 |                    0.633 |            0.567 |        0.065 |      0.014 |       0.12  |               1 |
| field      |         10 | Gothic                   |           12 |                    0.817 |            0.848 |       -0.031 |     -0.154 |       0.092 |               0 |
| field      |         11 | Art Deco                 |           22 |                    0.531 |            0.548 |       -0.017 |     -0.138 |       0.115 |               0 |
| field      |         12 | Early Modernism          |           76 |                    0.585 |            0.546 |        0.04  |     -0.035 |       0.113 |               0 |
