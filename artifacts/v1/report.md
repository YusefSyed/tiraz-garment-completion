# Tiraz annotation-only garment-category completion

This is category-set completion, not image recognition, taste, preference, or production recommendation quality.

Source-group partitions: `{"cal": 1678, "temp": 1728, "test": 962, "train": 27683, "tune": 3330}`.

| Model | Condition | Top-1 | Macro F1 (27) | Set coverage | Mean set size | Singleton rate |
|---|---|---:|---:|---:|---:|---:|
| frequency | test | 0.378 | 0.042 | 0.909 | 11.05 | 0.000 |
| frequency | stress | 0.353 | 0.037 | 0.878 | 10.95 | 0.000 |
| cooccurrence | test | 0.527 | 0.125 | 0.918 | 8.27 | 0.000 |
| cooccurrence | stress | 0.379 | 0.056 | 0.842 | 7.65 | 0.000 |
| set_encoder_seed_17 | test | 0.586 | 0.166 | 0.912 | 4.72 | 0.045 |
| set_encoder_seed_17 | stress | 0.367 | 0.070 | 0.556 | 3.73 | 0.049 |
| set_encoder_seed_29 | test | 0.585 | 0.165 | 0.911 | 4.65 | 0.040 |
| set_encoder_seed_29 | stress | 0.373 | 0.071 | 0.569 | 3.83 | 0.048 |
| set_encoder_seed_43 | test | 0.590 | 0.171 | 0.911 | 4.71 | 0.031 |
| set_encoder_seed_43 | stress | 0.369 | 0.070 | 0.579 | 3.88 | 0.006 |

## Interpretation

All three neural seeds are shown, including unfavorable outcomes. The frequency and co-occurrence baselines use the same training partition. Checkpoint selection uses tuning data; temperatures and conformal quantiles use disjoint calibration halves. No final-test tuning occurs.

The stress condition removes additional visible categories without changing the target or source-image unit. Its rows are paired observations, not extra independent samples. Conformal nominal coverage requires exchangeability; neither official validation nor context corruption guarantees it.

Inspect metrics.json for calibration, paired resampling intervals and singleton accuracy. Empty/small prediction sets are not silently coerced into confident recommendations. The existing deterministic app ranker is unchanged.

## Provenance

Fashionpedia Consortium annotations/ontology: CC BY 4.0. https://fashionpedia.github.io/home/data_license.html . No images downloaded. Deep Sets method: https://arxiv.org/abs/1703.06114 . Split conformal reference: https://arxiv.org/abs/2107.07511 . These methods are prior art; the contribution is this reproducible task, leakage controls and comparative audit.
