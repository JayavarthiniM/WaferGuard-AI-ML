# Wafer-GUARD diagnostics and limitations

This document holds supporting diagnostics that do not fit in the one-page audit. They are exploratory, and they do not establish production readiness.

## Chronological validation

The pipeline uses time-ordered train/validation/test partitions. Earlier experiments compared logistic regression, Extra Trees, Random Forest, threshold sensitivity, and five autoencoder seeds across earlier validation windows. Results vary substantially by window and model. For example, the Random Forest validation average precision varied from 0.0598 to 0.3252 in one set of diagnostic experiments; this variability argues against selecting a replacement based on one favorable fold.

## Threshold sensitivity

Maximizing F1 on a single 313-row validation window with only 11 positive examples selected a threshold near zero. That threshold flagged most wafers and produced many false alarms on the final test. Fixed-threshold checks on rolling validation windows showed that precision/recall tradeoffs are unstable across time windows. The final test set was not used to select these diagnostic thresholds.

A better operational policy should be specified before tuning: for example, a top-k alert budget, a minimum precision target, or a documented cost ratio for missed defects versus false alarms. Report precision@k and recall@k across chronological validation folds before freezing any policy. Do not use the final test set to select k or the decision rule.

## Seed stability

Five autoencoder seeds showed weak validation ranking performance. Validation average precision was approximately 0.0290–0.0324 and ROC-AUC approximately 0.3874–0.4455 in that experiment. This indicates that changing initialization alone did not resolve the problem.

## Temporal drift

The drift report compares observed distributions across chronological splits. It found changes in some sensor medians and missingness rates. These are descriptive associations, not proof of root cause. Sensor distribution shifts should be investigated using training-derived summaries and earlier validation windows; future/test statistics must not be fed back into the training preprocessing pipeline.

## Dataset size and uncertainty

The dataset contains 1,567 rows and 104 defects overall (6.64%). The final chronological test has 314 rows and 17 defects (5.41%). With so few positives, F1 and recall have substantial sampling uncertainty. Confidence intervals should be estimated by bootstrapping test rows only for reporting, not for model or threshold selection; with clustered or time-dependent wafers, block-based intervals may be more appropriate than an iid bootstrap.

## PS-L logic

The production pipeline infers legacy rows from `sensor_0 == 0.0`; it does not read the evaluator-only mask. The data audit found 146 sentinel rows and checked that their `sensor_1` values use the specified firmware code set. The code creates a legacy indicator and three code flags; it sets the sentinel and legacy code to missing before continuous preprocessing. A future audit could compare the inferred legacy flags against the evaluator-only mask strictly offline for sanity checking, but the mask and any comparison output must not be included in the submission.

## Reproducibility and limitations

- Run `python audit_checks.py` to validate source data assumptions and chronological split invariants.
- Run each diagnostic script from the repository root to regenerate its corresponding JSON artifact.
- Notebook metrics may vary slightly across library versions or nondeterministic hardware operations; record the results from the exact environment used for submission.
- The current model is not production-ready. No claim of F1 ≥ 0.90 is supported by the repaired chronological evaluation.


## Final-test uncertainty and alert-budget view

The notebook now saves test predictions and computes a seeded 2,000-replicate row-wise bootstrap for F1, precision, recall, ROC-AUC, and average precision. The resulting 95% percentile intervals are: F1 [0.068, 0.200], precision [0.036, 0.114], recall [0.500, 0.905], ROC-AUC [0.489, 0.750], and average precision [0.047, 0.157]. Because rows are time-ordered, this iid bootstrap may understate uncertainty if nearby wafers are dependent; it is reporting-only and never used for tuning.

A prespecified ranking view on the same final test ranks wafers by predicted probability: top 5% means k=16, precision@k=0 and recall@k=0; top 10% means k=32, precision@k=0.0625 and recall@k=0.1176. These test summaries are for transparent reporting only; do not use them to select a threshold, k, or model.
