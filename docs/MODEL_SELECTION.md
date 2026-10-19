# Validation-only model and threshold decision

## Protocol

This report uses the expanding-window folds in `artifacts/rolling_validation_metrics.json` and prespecified thresholds in `artifacts/threshold_sensitivity.json`. The final 20% chronological test period is excluded from these diagnostics. Average precision (AP) is emphasized because defect prevalence is low; ROC-AUC is supplementary. These folds are small and have different prevalence, so means are descriptive, not guarantees.

## Expanding-window comparison

Each fold's random-ranking AP baseline is its own defect prevalence, not a shared global number. The validation folds contain only 6, 9, and 11 defects respectively; therefore a small number of reordered positives can move AP substantially. The lift column below is AP divided by that fold's prevalence, shown as a descriptive ratio (not a significance test).

| Fold | Validation rows | Defects | Prevalence / random AP baseline |
|---|---:|---:|---:|
| 1 | 235 | 6 | 0.0255 (2.55%) |
| 2 | 235 | 9 | 0.0383 (3.83%) |
| 3 | 313 | 11 | 0.0351 (3.51%) |

| Model | Fold 1 AP | Fold 1 AP / baseline | Fold 2 AP | Fold 2 AP / baseline | Fold 3 AP | Fold 3 AP / baseline | Mean AP | Fold 1 AUC | Fold 2 AUC | Fold 3 AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic regression (balanced) | 0.0359 | 1.41x | 0.0957 | 2.50x | 0.0299 | 0.85x | 0.0539 | 0.5597 | 0.6618 | 0.4028 |
| Extra Trees | 0.0632 | 2.47x | 0.1798 | 4.70x | 0.0519 | 1.48x | 0.0983 | 0.7293 | 0.5860 | 0.6252 |
| Random Forest | 0.0838 | 3.28x | 0.3252 | 8.49x | 0.0598 | 1.70x | 0.1562 | 0.7067 | 0.6853 | 0.6632 |

Random Forest has the strongest mean AP among these tested baselines, but its AP ranges from 0.0598 to 0.3252 across folds. Fold 3 AP of 0.0598 is only 1.70x the fold-specific random baseline of 0.0351, and fold 3 contains only 11 defects. The large mean is strongly influenced by fold 2 (AP 0.3252; 8.49x baseline) and does not repeat at the same level in fold 3. These are small, noisy folds; this is evidence of temporal instability, not proof of robust predictive signal or evidence that the final test would improve. Do not select a model based on the final test period.

## Threshold stability

The prespecified Random Forest threshold of 0.10 behaved differently by fold:

- Fold 1: recall 0.667, precision 0.037, and 46.0% of rows flagged.
- Fold 2: recall 0.556, precision 0.143, and 14.9% of rows flagged.
- Fold 3: recall 0.000, precision 0.000, despite 7.0% of rows flagged.

At thresholds 0.30–0.50, these folds generally produced no positive predictions. Thus a single score threshold is not a robust operating policy here. Thresholds should be tied to an explicit alert budget / cost matrix and calibrated on multiple forward-in-time validation windows, not chosen to maximize a single small validation F1.

## Autoencoder seed sensitivity

Five training seeds on the fixed training/validation split produced validation AP from 0.0290 to 0.0324 and ROC-AUC around 0.39–0.45 (see `artifacts/seed_stability.json`). The selected thresholds ranged from approximately 1.59e-05 to 3.65e-05, and selected-threshold predictions flagged 253–281 of 313 validation rows in the recorded runs. This indicates unstable score scale and an unusably high alert burden for the F1-selected policy.

## Decision

1. Keep the repaired PyTorch notebook as the auditable correction of the supplied pipeline; do not present it as a successful production detector.
2. Treat Random Forest as a candidate baseline for further *train/validation-only* research, not as a confirmed replacement. Its fold-to-fold AP instability is material.
3. Do not claim the target F1 >= 0.90 is achieved. The measured final test result remains the recorded NO-GO result in `AUDIT.md`.
4. Before production, obtain more chronological data and defect examples, confirm what the timestamp means operationally, establish the alert budget and cost of false negatives/false positives, then conduct a pre-registered forward-chaining evaluation with a locked final period.

These findings are diagnostic. They do not establish causal validity of the source data or certify real-world wafer deployment readiness.
