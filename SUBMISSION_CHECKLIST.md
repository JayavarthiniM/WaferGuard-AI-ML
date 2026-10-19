# Pre-submission checklist

## Required files
- [x] `wafer_guard_fixed.ipynb`
- [x] `AUDIT.md`
- [x] `README.md`
- [x] `requirements.txt`
- [x] Dataset redistribution risk handled: CSV intentionally omitted from release archive; obtain official challenge data and place it beside the notebook

## Integrity review
- [x] Notebook parses and validates as nbformat v4.
- [x] Notebook contains no stored cell execution errors.
- [x] Recorded confusion matrix sums to 314 test rows.
- [x] Recorded test prevalence matches 17 / 314.
- [x] Audit documents preprocessing leakage, tensor broadcasting, and strict PS-L sentinel/category handling.
- [x] Three-seed ablation ran for C0–C7; raw results and interpretation are saved under `artifacts/ablation_results.json` and `docs/ABLATION_STUDY.md`.
- [x] Notebook re-executed after strict PS-L change; no stored execution errors.
- [x] Model performance is reported honestly; no claim of F1 >= 0.90.

- [x] Fixed-model diagnostics include separately labeled supervised and reconstruction-error scores, with AP upper-tail and two-sided AUC permutation tests (5,000 row-wise permutations each).
- [x] Diagnostic plots are saved; monthly test chart is explicitly limited to October 2008 only.
- [x] `.pytest_cache/` and `__pycache__/` are excluded from the release archive.

## Before submitting
- [ ] Run the notebook from a clean environment using the instructions in README.md.
- [ ] Run `python audit_checks.py` and retain the output.
- [ ] Review the final diff and ensure no evaluator-only mask file or other prohibited artifact is included.
- [ ] Publish these files in a public GitHub repository.
- [ ] Open the public repository in a logged-out/private browser to verify access.
- [ ] Submit the verified repository URL.

## Release status
**Code/audit package: prepared for review. Model release: NO-GO.** The recorded strict-PS-L model metrics are F1 0.1311, precision 0.0723, recall 0.7059, ROC-AUC 0.6243, and average precision 0.0841 on a 314-row chronological test set with 5.41% defect prevalence. Structural checks do not establish production readiness.
