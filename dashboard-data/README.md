# dashboard-data: causal-double-ml

Key results from `notebooks/analysis.ipynb` ("Does 401(k) eligibility raise
wealth?"), exported for the interactive dashboard. All numbers come from the
notebook's executed outputs and the repo's committed data files
(`data/main_results.json`, `data/mc_results.json`,
`data/sensitivity_results.json`). Simulated data with a known true effect of
$8,000 (see `data/build_data.py`).

## Files

- **estimates.json** — the four estimators head to head.
  Fields: `truth` (known true effect, $), `n`, `eligibility_rate`, `seed`,
  `methods[]`: `method`, `estimate` ($), `ci_low`/`ci_high` ($, 95% CI),
  `bias` (estimate minus truth, $), `covers_truth` (bool).
- **balance.json** — covariate balance before/after propensity-score matching.
  Fields: `matched_pairs`, `treated_dropped_by_caliper`,
  `covariates[]`: `covariate`, `smd_before`, `smd_after` (standardized mean
  differences; below 0.1 is well balanced).
- **sensitivity.json** — DML estimate under 6 learner/fold configurations.
  Fields: `truth`, `configs[]`: `config` ("learner, K=folds"), `estimate`,
  `ci_low`, `ci_high` ($).
- **monte_carlo.json** — bias and coverage over 30 fresh simulated datasets.
  Fields: `n_datasets`, `n_per_dataset`, `truth`,
  `methods[]`: `method`, `mean_bias` ($), `coverage_95ci` (fraction of the 30
  intervals covering the truth), `mean_ci_width` ($).
- **confounding.json** — group means by eligibility (the selection mechanism).
  Fields: `income_eligibility_correlation`,
  `groups[]`: `variable`, `not_eligible`, `eligible`, `difference` (rounded
  means; binary variables show as 0/1).
