"""
Estimators for the 401(k) eligibility effect on net financial assets.

Each estimator returns a dict with:
  estimate  : point estimate of the ATE (dollars)
  se        : standard error
  ci_low    : lower bound of the 95% confidence interval
  ci_high   : upper bound of the 95% confidence interval
  extra     : method-specific details (balance stats, learner info, ...)

1. naive_ols       -- Y on D only. Biased when D correlates with X (selection).
2. ols_controls    -- Y on D + X, linear. Biased when g(X) is nonlinear or
                      when residual imbalance remains. HC0 robust SEs.
3. propensity_match -- logistic propensity score, 1:1 nearest-neighbor
                      matching without replacement on the score, ATT estimate,
                      paired SE. Reports standardized mean differences.
4. dml             -- double machine learning with cross-fitting, implemented
                      manually (no econml):
                        * split sample into K folds
                        * on each fold, learn E[Y|X] and E[D|X] from the other
                          folds, predict out-of-fold
                        * Y_res = Y - m(X), D_res = D - e(X)
                        * theta_hat = (D_res' Y_res) / (D_res' D_res)
                        * heteroskedasticity-robust SE for that slope
                      The nuisance functions are learned with flexible ML, so
                      nonlinearity in g(X) and in the propensity is captured.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold

Z95 = 1.96


def _with_ci(estimate, se, extra=None):
    return {
        "estimate": float(estimate),
        "se": float(se),
        "ci_low": float(estimate - Z95 * se),
        "ci_high": float(estimate + Z95 * se),
        "extra": extra or {},
    }


def naive_ols(y, d):
    """Y on D only, HC0 robust SE."""
    X = np.column_stack([np.ones_like(d), d])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    bread = np.linalg.inv(X.T @ X)
    meat = (X * (resid ** 2)[:, None]).T @ X
    se = float(np.sqrt(np.diag(bread @ meat @ bread))[1])
    return _with_ci(beta[1], se, {"note": "HC0 robust SE"})


def ols_controls(y, d, x):
    """Y on D + X (linear), HC0 robust SE."""
    X = np.column_stack([np.ones_like(d), d, x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    bread = np.linalg.inv(X.T @ X)
    meat = (X * (resid ** 2)[:, None]).T @ X
    se = float(np.sqrt(np.diag(bread @ meat @ bread))[1])
    return _with_ci(beta[1], se, {"note": "HC0 robust SE"})


def _smd(x, d):
    """Standardized mean difference for each column of x (treated vs control)."""
    xt, xc = x[d == 1], x[d == 0]
    pooled = np.sqrt((xt.var(0, ddof=1) + xc.var(0, ddof=1)) / 2.0)
    pooled = np.where(pooled == 0, 1.0, pooled)
    return (xt.mean(0) - xc.mean(0)) / pooled


def propensity_match(y, d, x, col_names=None, caliper=None, with_replacement=True):
    """
    Logistic propensity score + 1:1 nearest-neighbor matching on the score.
    With replacement (default): every treated unit is matched to its closest
    control, controls may be reused. This is the standard choice when the
    control pool is much smaller than the treated pool, because matching
    without replacement exhausts the good controls and leaves covariate
    imbalance. A caliper drops treated units with no close control; the
    estimand is then the ATT on the matched treated units.

    ATT estimate with a paired t-type SE. The SE treats the matches as fixed
    and the propensity score as known, so it is stated as optimistic in the
    write-up.
    """
    ps_model = LogisticRegression(max_iter=1000, C=1.0)
    ps_model.fit(x, d)
    ps = ps_model.predict_proba(x)[:, 1]

    tr_idx = np.flatnonzero(d == 1)
    co_idx = np.flatnonzero(d == 0)
    if caliper is None:
        # standard default: 0.2 * SD of the logit propensity score
        logit = np.log(ps / (1.0 - ps))
        caliper = 0.2 * logit.std()
    nn = NearestNeighbors(n_neighbors=1).fit(ps[co_idx].reshape(-1, 1))
    dist, loc = nn.kneighbors(ps[tr_idx].reshape(-1, 1))
    keep = dist[:, 0] <= caliper
    matched_co = co_idx[loc[keep, 0]]
    matched_tr = tr_idx[keep]

    diffs = y[matched_tr] - y[matched_co]
    att = diffs.mean()
    se = diffs.std(ddof=1) / np.sqrt(len(diffs))

    smd_before = _smd(x, d)
    if with_replacement:
        # balance is assessed on the matched treated vs matched controls
        # (controls weighted by reuse count)
        reuse = np.bincount(matched_co, minlength=len(y)).astype(float)
        xt = x[matched_tr]
        wc = reuse[reuse > 0]
        xc = x[reuse > 0]
        wmean = (xc * wc[:, None]).sum(0) / wc.sum()
        wvar = (wc[:, None] * (xc - wmean) ** 2).sum(0) / wc.sum()
        pooled = np.sqrt((xt.var(0, ddof=1) + wvar) / 2.0)
        pooled = np.where(pooled == 0, 1.0, pooled)
        smd_after = (xt.mean(0) - wmean) / pooled
    else:
        d_matched = np.zeros(len(y), dtype=bool)
        d_matched[matched_tr] = True
        d_matched[matched_co] = True
        smd_after = _smd(x[d_matched], d[d_matched])
    names = col_names if col_names else [f"x{i}" for i in range(x.shape[1])]
    balance = {
        n: {"before": float(b), "after": float(a)}
        for n, b, a in zip(names, smd_before, smd_after)
    }
    extra = {
        "balance": balance,
        "n_treated": int(d.sum()),
        "n_matched_pairs": int(len(diffs)),
        "n_treated_dropped": int(len(tr_idx) - len(diffs)),
        "caliper": float(caliper),
        "with_replacement": bool(with_replacement),
        "mean_abs_smd_before": float(np.abs(smd_before).mean()),
        "mean_abs_smd_after": float(np.abs(smd_after).mean()),
    }
    return _with_ci(att, se, extra)


def dml(y, d, x, learner_y=None, learner_d=None, n_splits=5, seed=20261002):
    """
    Double machine learning with manual K-fold cross-fitting.

    theta_hat solves sum_i (y_i - m(x_i) - theta (d_i - e(x_i))) (d_i - e(x_i)) = 0
    with m, e fitted out-of-fold. Robust SE:
      Var(theta_hat) = sum_i v_i^2 r_i^2 / (sum_i v_i^2)^2,
    where v = d residuals, r = y residuals minus theta_hat * v.
    """
    if learner_y is None:
        learner_y = RandomForestRegressor(
            n_estimators=200, min_samples_leaf=10, n_jobs=-1, random_state=seed)
    if learner_d is None:
        learner_d = RandomForestRegressor(
            n_estimators=200, min_samples_leaf=10, n_jobs=-1, random_state=seed)
    y_res = np.zeros_like(y, dtype=float)
    d_res = np.zeros_like(d, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in kf.split(x):
        ly = learner_y.__class__(**learner_y.get_params())
        ld = learner_d.__class__(**learner_d.get_params())
        ly.fit(x[tr], y[tr])
        ld.fit(x[tr], d[tr])
        y_res[te] = y[te] - ly.predict(x[te])
        d_res[te] = d[te] - ld.predict(x[te])
    theta = float((d_res @ y_res) / (d_res @ d_res))
    r = y_res - theta * d_res
    var = float(np.sum((d_res ** 2) * (r ** 2)) / (np.sum(d_res ** 2) ** 2))
    se = float(np.sqrt(var))
    extra = {
        "learner_y": type(learner_y).__name__,
        "learner_d": type(learner_d).__name__,
        "n_splits": n_splits,
        "note": "heteroskedasticity-robust SE",
    }
    return _with_ci(theta, se, extra)


def gbm_learner(seed=20261002):
    return GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                    random_state=seed)


def rf_learner(seed=20261002):
    return RandomForestRegressor(n_estimators=200, min_samples_leaf=10,
                                 n_jobs=-1, random_state=seed)
