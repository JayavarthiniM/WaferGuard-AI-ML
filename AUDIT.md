# Wafer-GUARD audit

## Required findings

**1. Data leakage — fixed.** The original notebook fitted imputation/scaling on all rows and applied SMOTE before splitting. Holdout statistics influenced preprocessing, and synthetic minority neighbours could appear across train/test. The repair sorts by timestamp, uses the earliest 60% / next 20% / latest 20% for train/validation/test, fits preprocessing on train only, and does not resample validation or test.

**2. Corrupted reconstruction loss — fixed.** The original input `[B, 1, F]` and decoder output `[B, F]` broadcast during subtraction to `[B, B, F]`. With batch 64 and 590 sensors this is `torch.Size([64, 64, 590])`, comparing each reconstruction against every batch input. The repaired notebook keeps input/reconstruction at `[B, F]`, logits/labels at `[B]`, and asserts matching shapes.

**3. PS-L invariant blindness — fixed.** `sensor_0 == 0.0` identifies the legacy sentinel; on legacy rows, `sensor_1` values 100/200/300 are categorical codes. The notebook creates four indicators, neutralizes the two slots after scaling, and masks them out of reconstruction loss. Thus fabricated medians are not model inputs or reconstruction targets. It infers 146 legacy rows and never reads the evaluator-only mask.

## Measured metrics

| Run | Protocol | F1 | Precision | Recall | ROC-AUC | AP |
|---|---|---:|---:|---:|---:|---:|
| Original reproduced | Random split; full-data fit; SMOTE before split; broadcast bug | 0.9915 | 0.9832 | 1.0000 | 1.0000 | 1.0000 |
| Repaired (strict PS-L) | Chronological 60/20/20; train-only fit; no evaluation resampling; aligned tensors; masked PS-L slots | 0.1311 | 0.0723 | 0.7059 | 0.6243 | 0.0841 |

The original result was reproduced with only the install cell skipped; F1=0.9915 is contaminated and not a valid future-performance estimate. The repaired test contains 314 wafers and 17 defects (5.41% prevalence); overall dataset prevalence is 6.64%. Confusion matrix TN=143, FP=154, FN=5, TP=12. A 2,000-replicate iid row bootstrap gives 95% intervals: F1 [0.068, 0.199], precision [0.036, 0.113], recall [0.500, 0.905], ROC-AUC [0.490, 0.751], AP [0.047, 0.158]. These intervals ignore temporal dependence. AP's interval overlaps test prevalence; useful ranking signal is not demonstrated.

**Classifier score and threshold.** The decision score is the supervised classification head probability (`sigmoid(logits)`); reconstruction loss is an auxiliary training objective, not the deployed anomaly score. At 5.4% test prevalence, average precision (AP), compared with prevalence, is the primary ranking metric. The threshold procedure is leakage-free in the sense that it is selected on validation only and frozen before test scoring, but the selected operating point is degenerate: threshold approximately `4.08e-5` flags 166/314 wafers (52.9%) to catch 12/17 defects. That yields precision 7.23%, recall 70.59%, F1 13.11%, and 154 false alarms. This is not a viable alerting threshold. Top-5% and top-10% ranked alerts catch 0/17 defects in the final test.

**Pooled out-of-time check.** Three expanding-window supervised-autoencoder folds, all before the final 20% holdout, pool 470 predictions with 20 defects (4.26% prevalence). Pooled AP is 0.0407 and ROC-AUC 0.4744; a contiguous-block bootstrap gives AP 95% interval [0.0213, 0.0730]. Top-5% and top-10% alert budgets catch no defects. The pooled ROC-AUC of 0.4744 is below chance. Together with validation AUC values around 0.39–0.45 in the seed-stability check and changing fold prevalence, this indicates no stable ranking signal across time. These rolling windows are supporting diagnostics, not a replacement final test. Details: `artifacts/pooled_oot_metrics.json`.

**Ablations.** Three-seed results for all eight configurations are in `docs/ABLATION_STUDY.md` and `artifacts/ablation_results.json`. C0→C1 bundles protocol changes and is not a causal estimate for one bug. C6 and strict C7 are practically indistinguishable; C7 is retained because it avoids fabricated legacy reconstruction targets.

## Release decision: NO-GO

The model misses the F1≥0.90 target, flags over half the final test as defective, and has no demonstrated useful ranking signal. Do not tune on the final test or claim production readiness. See `docs/DIAGNOSTICS.md` for supporting evidence. Run invariant checks with `python -m pytest -q`, reproduce the data checks with `python audit_checks.py`, and run all cells in `wafer_guard_fixed.ipynb`; fresh Colab verification and public-repository publishing remain submission steps.

**Fixed-model score diagnostics (descriptive; no retuning).** The supervised head uses `sigmoid(logits)`; reconstruction error is an auxiliary score. On the final test, supervised AP/AUC are 0.0841/0.6243. Reconstruction-error AP/AUC are recorded in `artifacts/fixed_model_diagnostics.json`; higher error is the pre-specified anomaly direction, and the observed AUC of 0.3167 is inverse to that direction. For each score, the JSON reports the observed AP/AUC, AP upper-tail permutation p-value, and two-sided AUC p-value versus 0.5, each from 5,000 row-wise label permutations. These tests ignore temporal dependence and are descriptive only. The score direction was not flipped after inspecting test results.

**Plot limitation.** The final test timestamps all fall in **October 2008** (314 rows), so the test-period monthly defect-rate chart contains only one month and cannot establish a month-to-month trend. Use earlier-period data for descriptive temporal plots, while keeping all model selection strictly before the final holdout.
