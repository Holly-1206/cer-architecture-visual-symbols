# Inter-annotator agreement (C7)

Blind re-annotation of 200 images (reference 98, field 54, historical 48).

| statistic                                |   value |   ci95_low |   ci95_high |
|:-----------------------------------------|--------:|-----------:|------------:|
| lineage_macro_kappa                      |   0.401 |      0.331 |       0.464 |
| lineage_pooled_kappa                     |   0.408 |      0.345 |       0.476 |
| lineage_set_exact_match                  |   0.355 |      0.29  |       0.42  |
| lineage_set_mean_jaccard                 |   0.437 |      0.373 |       0.496 |
| lineage_krippendorff_alpha               |   0.409 |      0.339 |       0.47  |
| symbol_pooled_kappa                      |   0.33  |      0.291 |       0.367 |
| symbol_krippendorff_alpha                |   0.328 |      0.291 |       0.365 |
| symbol_median_kappa_frequent(>=10 marks) |   0.291 |    nan     |     nan     |

|   style_id | style_name_en            |   n_original_positive |   n_reviewer_positive |   percent_agreement |   cohen_kappa |   ci95_low |   ci95_high |
|-----------:|:-------------------------|----------------------:|----------------------:|--------------------:|--------------:|-----------:|------------:|
|         01 | Russian Traditional      |                    21 |                    19 |               0.89  |         0.389 |      0.174 |       0.575 |
|         02 | Russian Timber Dwelling  |                    13 |                    10 |               0.915 |         0.217 |     -0.039 |       0.447 |
|         03 | Russian Railway          |                    40 |                    52 |               0.82  |         0.494 |      0.349 |       0.623 |
|         04 | Byzantine / Orthodox     |                    17 |                    18 |               0.925 |         0.53  |      0.297 |       0.732 |
|         05 | Eclecticism              |                    22 |                    32 |               0.8   |         0.148 |     -0.019 |       0.325 |
|         06 | Art Nouveau              |                    23 |                    13 |               0.9   |         0.394 |      0.167 |       0.593 |
|         07 | Baroque                  |                    16 |                    15 |               0.915 |         0.406 |      0.158 |       0.613 |
|         08 | Chinese Baroque          |                    27 |                    12 |               0.915 |         0.525 |      0.305 |       0.696 |
|         09 | Classical / Neoclassical |                    15 |                    22 |               0.875 |         0.258 |      0.036 |       0.446 |
|         10 | Gothic                   |                    13 |                    11 |               0.95  |         0.557 |      0.273 |       0.769 |
|         11 | Art Deco                 |                    16 |                    15 |               0.955 |         0.685 |      0.445 |       0.856 |
|         12 | Early Modernism          |                    20 |                    21 |               0.855 |         0.212 |      0.016 |       0.404 |

## Draft sentence for Section 3.3

To measure agreement, the reviewer re-annotated a stratified random sample of 200 images without seeing the original labels. Agreement on lineage attribution was κ = 0.41 (95% CI 0.34–0.48) over all image–lineage decisions (mean per-lineage κ = 0.40; per-lineage range 0.15–0.69), and on the presence of A–D symbols κ = 0.33 (95% CI 0.29–0.37).