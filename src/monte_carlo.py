"""
Monte Carlo study: bias, coverage, and interval width for each estimator.

Repeats the full analysis on R fresh simulated datasets (n=4000 each, same
DGP as data/build_data.py, seeds 20261002+1 .. +R) and records, per method:
  mean_bias   : average (estimate - true effect)
  coverage    : share of 95% CIs that contain the true effect
  mean_width  : average CI width
A well-calibrated method has coverage near 0.95.

Run: python3 src/monte_carlo.py   -> writes data/mc_results.json
"""

import sys, json, time
import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "data")
from build_data import generate, TAU_TRUE
from estimators import naive_ols, ols_controls, propensity_match, dml, rf_learner

R = 30
N = 4000
COLS = ["inc", "age", "educ", "fsize", "marr", "pira", "hown"]
# fewer trees keeps the study fast; the main analysis uses 200
FAST = 100


def rf_fast(seed):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=FAST, min_samples_leaf=10,
                                 n_jobs=-1, random_state=seed)


def main():
    methods = {
        "Naive OLS": lambda y, d, x, s: naive_ols(y, d),
        "OLS with controls": lambda y, d, x, s: ols_controls(y, d, x),
        "Propensity-score matching": lambda y, d, x, s: propensity_match(y, d, x, COLS),
        "DML (K=5, random forest)": lambda y, d, x, s: dml(y, d, x, rf_fast(s), rf_fast(s),
                                                           n_splits=5, seed=s),
    }
    stats = {m: {"bias": [], "cover": [], "width": []} for m in methods}
    t0 = time.time()
    for rep in range(R):
        seed = 20261002 + 1 + rep
        df = generate(n=N, seed=seed)
        y = df["nettfa"].values
        d = df["elig401k"].values
        x = df[COLS].values
        for name, fn in methods.items():
            r = fn(y, d, x, seed)
            stats[name]["bias"].append(r["estimate"] - TAU_TRUE)
            stats[name]["cover"].append(r["ci_low"] <= TAU_TRUE <= r["ci_high"])
            stats[name]["width"].append(r["ci_high"] - r["ci_low"])
        print(f"replicate {rep + 1}/{R} done ({time.time() - t0:.0f}s)", flush=True)
    out = []
    for name, s in stats.items():
        out.append({
            "Method": name,
            "Mean bias ($)": float(np.mean(s["bias"])),
            "Coverage of 95% CI": float(np.mean(s["cover"])),
            "Mean CI width ($)": float(np.mean(s["width"])),
        })
    with open("data/mc_results.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
