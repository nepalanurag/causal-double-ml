# Does 401(k) Eligibility Raise Wealth? A Four-Estimator Comparison with Double Machine Learning

## 1. Background

Whether employer-sponsored retirement plans cause households to accumulate more wealth is one of the standard examples in applied econometrics (Wooldridge, 2013, Example 7.12 and Chapter 21). The policy question matters: if eligibility itself raises saving, expanding access is worthwhile; if eligible workers were simply going to save anyway, the policy lever is weaker.

The statistical problem is selection. In observational data, 401(k) eligibility is not assigned at random. It is offered disproportionately to higher-income, older, more educated workers at larger employers, and those same traits predict wealth regardless of any retirement plan. Any comparison that does not account for this will mistake the traits of the eligible for the effect of eligibility.

This project compares four estimators on the same data: naive OLS, OLS with linear controls, propensity-score matching, and double machine learning (DML) with cross-fitting. The comparison is run on simulated data with a known true effect, so each method can be scored on bias and interval coverage rather than argued about in the abstract.

## 2. Data

The dataset is **simulated** (n = 8,000). The classical Wooldridge 401k file was sought from four public CSV mirrors on 2026-10-02; all four attempts failed (three 404s, one HTML page served instead of data; the URLs are listed in `data/build_data.py`). Since no CSV source was reachable and `.dta` readers were not installed, the data were generated from a documented process with a fixed seed (20261002) and a fixed true average treatment effect of $8,000.

### 2.1 Variables

| Variable | Meaning | Construction |
|---|---|---|
| nettfa | Net financial assets ($) | outcome Y |
| elig401k | Eligible for 401(k) (0/1) | treatment D |
| inc | Household income ($) | lognormal, median ~$60k |
| age | Age in years | uniform 25-64 |
| educ | Years of schooling | normal, mean 14 |
| fsize | Family size | Poisson-based, 1-6 |
| marr | Married (0/1) | Bernoulli 0.65 |
| pira | Holds an IRA (0/1) | Bernoulli 0.35 |
| hown | Owns home (0/1) | Bernoulli 0.60 |

### 2.2 Data-generating process

1. Draw confounders X = (inc, age, educ, fsize, marr, pira, hown) from the distributions above.
2. Treatment: logit P(D=1|X) = -3.10 + 0.035*inc_k + 0.022*(age-40) + 0.12*educ + 0.35*marr + 0.60*pira + 0.25*hown + noise(0, 0.3), clipped to [0.02, 0.98]. Eligibility rate in the sample: 77%.
3. Outcome: Y = g(X) + 8000*D + noise(0, 45000), where g(X) = -60000 + 700*inc_k + 14000*sqrt(inc_k) + 800*(age-40) + 4000*(educ-14) - 3000*fsize + 8000*marr + 6000*pira + 10000*hown, with inc_k in thousands.

Two design choices deserve explanation. First, the income effect on wealth is concave (the square-root term): the first dollars of income build more wealth than later dollars. A linear specification in income therefore cannot fully absorb income confounding, which is exactly the realistic failure mode we want OLS-with-controls to exhibit. Second, the treatment effect is constant at $8,000, so ATE = ATT = $8,000 and every method targets the same number.

### 2.3 Confounding, shown empirically

| Group | n | Mean income | Mean age | Mean educ | Mean nettfa |
|---|---|---|---|---|---|
| Not eligible | 1,844 | $46,439 | 43.0 | 14.0 | $70,466 |
| Eligible | 6,156 | $79,209 | 45.0 | 14.0 | $133,550 |

The raw wealth gap is $63,085. Eligible workers earn 71% more on average (correlation between income and eligibility: 0.29) and are older. Eligibility rises monotonically with income decile (see `figures/confounding.png`). This is the selection that the naive estimator mistakes for a treatment effect.

## 3. Methods

All four estimators target E[Y(1) - Y(0)], the average causal effect of eligibility on net financial assets, under the usual identifying assumptions: unconfoundedness (D independent of potential outcomes given X) and overlap (0 < P(D=1|X) < 1). Both hold by construction in the simulation.

**Naive OLS.** Y regressed on D only. Consistent only under random assignment. Included as the baseline that shows the size of the selection problem. HC0 robust standard errors.

**OLS with controls.** Y regressed on D and all seven confounders linearly. Consistent if the conditional expectation of Y given (D, X) is truly linear. It is not (concave income effect), so residual confounding is expected. HC0 robust standard errors.

**Propensity-score matching.** Step 1: logistic regression of D on X gives each worker a propensity score. Step 2: each eligible worker is matched 1:1 to the nearest non-eligible worker on the score, with replacement and a caliper of 0.2 standard deviations of the logit score (Rosenbaum and Rubin, 1985). With replacement is the right choice here because the control pool (23% of the sample) is small: matching without replacement exhausted the good controls and left income with a standardized mean difference of 0.71. With replacement plus caliper, every covariate's absolute SMD fell below 0.07. The ATT is the mean within-pair wealth difference; the standard error is the paired t-type SE, which treats matches as fixed and the score as known (optimistic; stated as a limitation).

**Double machine learning (Chernozhukov et al., 2018).** Implemented manually with scikit-learn, no econml. The sample is split into K folds. For each fold, E[Y|X] and E[D|X] are learned on the other folds with random forests (200 trees, min_samples_leaf=10), predictions are made out-of-fold, and residuals are formed: Y~ = Y - m(X), D~ = D - e(X). The effect is the slope of Y~ on D~, i.e. theta = (D~'Y~)/(D~'D~), with heteroskedasticity-robust SE = sqrt(sum(D~^2 r^2)) / sum(D~^2), r = Y~ - theta*D~. Cross-fitting matters: predicting out-of-fold stops the forests from fitting the noise and leaking it into the estimate (overfitting the nuisance functions is the failure mode that naive "ML then regress" approaches suffer). Sensitivity is checked over learners (random forest vs gradient boosting) and K = 2, 5, 10.

## 4. Results

### 4.1 Headline comparison (n = 8,000; true effect $8,000)

| Method | Estimate | SE | 95% CI | Bias | CI covers $8,000 |
|---|---|---|---|---|---|
| Naive OLS | $63,085 | $1,767 | [$59,621, $66,549] | +$55,085 | No |
| OLS with controls | $11,542 | $1,294 | [$9,006, $14,077] | +$3,542 | No |
| Propensity-score matching | $6,287 | $894 | [$4,536, $8,039] | -$1,713 | Yes |
| DML (K=5, random forest) | $7,768 | $1,305 | [$5,210, $10,326] | -$232 | Yes |

![Forest plot](../figures/forest_plot.png)

### 4.2 Interpretation

The naive estimate overstates the effect by a factor of eight. Controlling linearly for the confounders removes the large majority of the bias but leaves +$3,542 (2.7 standard errors): the concave income effect leaks through the linear specification, exactly as the DGP was designed to demonstrate. Matching, once balance is genuinely achieved, and DML both recover the effect with intervals that cover the truth. DML has the smallest bias (-$232, under 0.2 SE).

### 4.3 Matching balance

Standardized mean differences, before vs after matching (with replacement + caliper; 6,156 matched pairs, 0 treated dropped):

| Covariate | Before | After |
|---|---|---|
| Income | +0.843 | -0.001 |
| Age | +0.234 | -0.051 |
| Education | +0.169 | +0.013 |
| Family size | +0.006 | +0.061 |
| Married | +0.200 | +0.039 |
| Has IRA | +0.267 | +0.056 |
| Owns home | +0.099 | -0.027 |

All post-match values are below the 0.1 rule of thumb. The love plot is in `figures/balance_plot.png`.

### 4.4 DML sensitivity

| Configuration | Estimate | 95% CI |
|---|---|---|
| Random forest, K=2 | $8,208 | [$5,636, $10,780] |
| Random forest, K=5 | $7,768 | [$5,210, $10,326] |
| Random forest, K=10 | $7,695 | [$5,144, $10,245] |
| Gradient boosting, K=2 | $7,570 | [$4,997, $10,144] |
| Gradient boosting, K=5 | $7,602 | [$5,031, $10,174] |
| Gradient boosting, K=10 | $7,527 | [$4,951, $10,102] |

The six configurations span $7,527-$8,208; every interval covers the true $8,000. The conclusion does not hinge on the learner or the fold count (see `figures/sensitivity.png`).

### 4.5 Monte Carlo: bias and coverage

One dataset is one draw, so single-interval coverage is anecdotal. The full analysis was repeated on 30 fresh datasets (n = 4,000 each, same DGP, seeds 20261003..20261032; DML with 100-tree forests for speed):

| Method | Mean bias ($) | Coverage of 95% CI | Mean CI width ($) |
|---|---|---|---|
| Naive OLS | +56,405 | 0.00 | 9,889 |
| OLS with controls | +4,246 | 0.30 | 7,182 |
| Propensity-score matching | +623 | 0.43 | 4,744 |
| DML (K=5, random forest) | +811 | 0.90 | 7,358 |

(Exact values in `data/mc_results.json`.) The naive interval never covers the truth. The linear-controls interval covers 30% of the time despite looking precise: a bias of $4,246 against a standard error near $1,800 puts the truth outside the interval most draws. Matching is nearly unbiased on average (+$623) but its intervals cover only 43%: the paired standard error treats reused controls as independent observations, so it understates the real sampling variability. This is a known weakness of the simple paired SE for matching with replacement. DML is the only method whose intervals cover at roughly the nominal rate (90%), with mean bias +$811.

## 5. Limitations and threats to validity

1. **Unconfoundedness is an assumption, not a finding.** All four methods assume every variable driving both eligibility and wealth was measured. An unmeasured confounder (financial literacy, employer generosity, risk tolerance) would bias all four estimates, and no statistical adjustment on these covariates can repair that.
2. **Simulated data.** The DGP is realistic but invented; the quantitative results describe the simulation, not the US labor market. The value of the exercise is methodological: with the truth known, the ranking of methods is earned, not asserted.
3. **Matching standard errors are optimistic**, as noted in Section 3; a bootstrap or Abadie-Imbens variance would be the next step.
4. **Constant treatment effect.** Real effects are heterogeneous; the DGP fixes the effect at $8,000 for comparability.
5. **Overlap is guaranteed by construction** (propensities clipped to [0.02, 0.98]). In real data, limited overlap would require trimming and would change the estimand.

## 6. References

- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., and Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1-C68.
- Rosenbaum, P. R. and Rubin, D. B. (1983). The central role of the propensity score in observational studies for causal effects. *Biometrika*, 70(1), 41-55.
- Rosenbaum, P. R. and Rubin, D. B. (1985). Constructing a control group using multivariate matched sampling methods that incorporate the propensity score. *The American Statistician*, 39(1), 33-38.
- Wooldridge, J. M. (2013). *Introductory Econometrics: A Modern Approach*, 5th ed. Cengage Learning. (The 401k example: Ch. 7 and Ch. 21.)

## 7. Reproducibility

Seed 20261002 is fixed in `data/build_data.py`, `src/estimators.py` (DML folds and forests), and `src/monte_carlo.py`. `python3 data/build_data.py` regenerates `data/401k_simulated.csv` byte-identically. The executed notebook `notebooks/analysis.ipynb` contains all outputs. Package versions used: Python 3.12, numpy, pandas, scikit-learn 1.9.1, scipy, matplotlib.
