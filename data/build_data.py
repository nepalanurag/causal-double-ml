"""
Data builder for the causal-double-ml project.

What this does
--------------
The question is the effect of 401(k) eligibility (treatment D) on net
financial assets (outcome Y) given confounders X (income, age, education,
family size, marital status, IRA participation, home ownership).

The classic teaching dataset for this question is the Wooldridge 401k
dataset. We tried to fetch a CSV mirror of it before simulating:

  1. https://raw.githubusercontent.com/wooldridgem/data/master/401k.csv  -> 404
  2. https://raw.githubusercontent.com/stanleydesu/wrapped/main/401k.csv -> 404
  3. https://raw.githubusercontent.com/wooldridgem/data/main/401k.csv    -> 404
  4. https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/401k.csv   -> HTML page, not data

(checked 2026-10-02; pyreadstat is not installed, so .dta files were not an option.)

Every attempt failed, so the dataset used in this project is SIMULATED.
The data-generating process below is documented in full, and the true
average treatment effect is fixed at TAU_TRUE = 8000 dollars. Because the
truth is known, every estimator can be judged on bias and confidence
interval coverage.

Usage:  python3 data/build_data.py   -> writes data/401k_simulated.csv
"""

import numpy as np
import pandas as pd

SEED = 20261002
N = 8000

# The true causal effect of 401(k) eligibility on net financial assets.
TAU_TRUE = 8000.0


def generate(n=N, seed=SEED):
    """Generate one observational dataset from the documented DGP."""
    rng = np.random.default_rng(seed)

    # --- Confounders X -------------------------------------------------
    # income_k: annual household income in thousands of dollars
    income_k = np.clip(rng.lognormal(np.log(60.0), 0.6, n), 10.0, 400.0)
    # age: years
    age = rng.uniform(25.0, 64.0, n)
    # educ: years of schooling
    educ = np.clip(np.round(rng.normal(14.0, 2.2, n)), 10.0, 20.0)
    # fsize: family size
    fsize = np.clip(rng.poisson(1.6, n) + 1, 1, 6).astype(float)
    # marr: married
    marr = rng.binomial(1, 0.65, n).astype(float)
    # pira: participates in an individual retirement account
    pira = rng.binomial(1, 0.35, n).astype(float)
    # hown: owns the home
    hown = rng.binomial(1, 0.60, n).astype(float)

    X = np.column_stack([income_k, age, educ, fsize, marr, pira, hown])

    # --- Treatment: 401(k) eligibility ---------------------------------
    # Selection mechanism: higher income, older, more educated, married,
    # and IRA-holding workers are more likely to be offered a 401(k).
    # This is the confounding: X drives both D and Y.
    logit_p = (
        -3.10
        + 0.035 * income_k
        + 0.022 * (age - 40.0)
        + 0.12 * educ
        + 0.35 * marr
        + 0.60 * pira
        + 0.25 * hown
        + rng.normal(0.0, 0.30, n)
    )
    p = 1.0 / (1.0 + np.exp(-logit_p))
    p = np.clip(p, 0.02, 0.98)          # keeps overlap: nobody is certain
    d = rng.binomial(1, p).astype(float)

    # --- Outcome: net financial assets -----------------------------------
    # g(X) is deliberately NONLINEAR in X: the income effect is concave
    # (diminishing returns, captured here with a square root), so a linear
    # OLS specification in income leaves real confounding on the table.
    # Flexible machine learners can pick this up; a linear model cannot.
    base = (
        -60000.0
        + 700.0 * income_k
        + 14000.0 * np.sqrt(income_k)
        + 800.0 * (age - 40.0)
        + 4000.0 * (educ - 14.0)
        - 3000.0 * fsize
        + 8000.0 * marr
        + 6000.0 * pira
        + 10000.0 * hown
    )
    y = base + TAU_TRUE * d + rng.normal(0.0, 45000.0, n)

    df = pd.DataFrame({
        "nettfa": y,          # net financial assets, dollars
        "elig401k": d,        # 1 = eligible for 401(k), 0 = not
        "inc": income_k * 1000.0,  # income in dollars
        "age": age,
        "educ": educ,
        "fsize": fsize,
        "marr": marr,
        "pira": pira,
        "hown": hown,
        "propensity_true": p,
    })
    return df


def main():
    df = generate()
    out = __file__.replace("build_data.py", "401k_simulated.csv")
    df.drop(columns=["propensity_true"]).to_csv(out, index=False)
    print(f"wrote {out}: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"true ATE = ${TAU_TRUE:,.0f}")
    print(f"eligibility rate = {df['elig401k'].mean():.3f}")


if __name__ == "__main__":
    main()
