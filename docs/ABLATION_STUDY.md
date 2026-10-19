# Controlled ablation study

## Protocol and caveats

`ablation_study.py` was executed with seeds **1337, 17, and 41**, 30 epochs per run. The JSON source of record is `artifacts/ablation_results.json`. Values below are mean ± population standard deviation across the three seeds. F1 thresholds are selected on validation only for C1–C7; C0 reproduces the contaminated original protocol with threshold 0.5 and a balanced holdout.

**Important attribution limitation:** C0 → C1 is a protocol bundle, not a one-variable experiment. It removes SMOTE-before-split, switches to a natural-prevalence split, and uses a validation-selected threshold. Its score collapse demonstrates that the original headline score is not trustworthy, but does not isolate the contribution of any one of those changes. C1 → C2 changes preprocessing fit scope; C2 → C3 changes random to chronological splitting; subsequent transitions target the shape fix, PS-L handling, imputation strategy, and strict PS-L masking respectively. The same neural architecture and training budget are used for all configurations, and the original broadcast bug is intentionally retained only where specified.

## Results

| Config | Test prevalence | F1 mean ± SD | AP mean ± SD | ROC-AUC mean ± SD |
|---|---:|---:|---:|---:|
| C0 original contaminated protocol | 50.0% | 0.988 ± 0.004 | 0.999 ± 0.001 | 0.999 ± 0.001 |
| C1 split first / no SMOTE | 6.7% | 0.171 ± 0.008 | 0.149 ± 0.022 | 0.671 ± 0.009 |
| C2 train-only preprocessing | 6.7% | 0.220 ± 0.007 | 0.148 ± 0.009 | 0.681 ± 0.016 |
| C3 chronological split | 5.4% | 0.152 ± 0.018 | 0.095 ± 0.013 | 0.626 ± 0.055 |
| C4 shape fix | 5.4% | 0.150 ± 0.013 | 0.114 ± 0.001 | 0.707 ± 0.003 |
| C5 PS-L indicators / missing codes | 5.4% | 0.130 ± 0.012 | 0.089 ± 0.006 | 0.645 ± 0.015 |
| C6 median imputation | 5.4% | 0.136 ± 0.010 | 0.090 ± 0.003 | 0.646 ± 0.013 |
| C7 strict PS-L neutral slots + masked reconstruction | 5.4% | 0.136 ± 0.010 | 0.089 ± 0.003 | 0.646 ± 0.014 |

## Interpretation

- C0’s near-perfect score is not a valid estimate of future performance. Its test prevalence is 50%, because SMOTE was run before the split; it also retains full-data preprocessing and the broadcast bug.
- Random split C1/C2 scores are not directly comparable to chronological C3–C7 scores because the evaluation periods differ. C2 → C3 is the more relevant signal about temporal generalization: all three metrics decrease on the later period.
- The shape fix raises AP and ROC-AUC relative to C3 in this experiment, although F1 changes little. The PS-L feature handling reduces the metrics in this model/configuration; correctness is not optional and should not be rejected merely because the score is lower.
- C6 and C7 are practically indistinguishable at this three-seed resolution. C7 is retained because it follows the stricter interpretation of the brief: legacy sentinel/code slots are neutralized and excluded from reconstruction targets rather than reconstructed toward fitted median values.
- The final notebook's single-seed held-out metrics are not the same estimand as these three-seed ablation means. After the strict PS-L update, the freshly executed notebook reports F1 0.1311, AP 0.0841, ROC-AUC 0.6243 on the fixed chronological test period (314 rows, 17 defects).

These small-seed results are descriptive, not statistical proof. They do not establish a production-useful signal. The final model remains **NO-GO**.
