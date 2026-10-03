"""Publication-style figures for the causal-double-ml project."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})


def forest_plot(results, true_value, path):
    """results: list of (name, estimate, ci_low, ci_high)."""
    names = [r[0] for r in results][::-1]
    est = np.array([r[1] for r in results])[::-1]
    lo = np.array([r[2] for r in results])[::-1]
    hi = np.array([r[3] for r in results])[::-1]
    ypos = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.errorbar(est, ypos, xerr=[est - lo, hi - est], fmt="o", capsize=5,
                color="#1f4e79", ecolor="#1f4e79", elinewidth=2, markersize=7)
    ax.axvline(true_value, color="#c0392b", linestyle="--", linewidth=1.5,
               label=f"True effect (${true_value:,.0f})")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Estimated effect of 401(k) eligibility on net financial assets ($)")
    ax.set_title("Treatment effect estimates with 95% confidence intervals")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def balance_plot(balance, path):
    """Love plot: standardized mean differences before/after matching."""
    names = list(balance.keys())
    before = [abs(balance[n]["before"]) for n in names]
    after = [abs(balance[n]["after"]) for n in names]
    ypos = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.scatter(before, ypos, s=70, label="Before matching", color="#c0392b",
               marker="o", zorder=3)
    ax.scatter(after, ypos, s=70, label="After matching", color="#1f4e79",
               marker="s", zorder=3)
    for i in ypos:
        ax.plot([before[i], after[i]], [i, i], color="gray", linewidth=1.2,
                alpha=0.6)
    ax.axvline(0.1, color="black", linestyle="--", linewidth=1,
               label="0.1 rule-of-thumb")
    ax.set_yticks(ypos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Absolute standardized mean difference")
    ax.set_title("Covariate balance before and after propensity-score matching")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def confounding_plot(df, path):
    """Show the confounding empirically: income distributions by eligibility."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    e1 = df.loc[df.elig401k == 1, "inc"]
    e0 = df.loc[df.elig401k == 0, "inc"]
    axes[0].hist(e0, bins=40, alpha=0.7, label="Not eligible", color="#c0392b")
    axes[0].hist(e1, bins=40, alpha=0.7, label="Eligible", color="#1f4e79")
    axes[0].set_xlabel("Household income ($)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Income distribution by 401(k) eligibility")
    axes[0].legend()

    bins = np.quantile(df["inc"], np.linspace(0, 1, 11))
    idx = np.clip(np.digitize(df["inc"], bins) - 1, 0, 9)
    mid = 0.5 * (bins[:-1] + bins[1:])
    rate = [df.loc[idx == k, "elig401k"].mean() for k in range(10)]
    nbin = [int((idx == k).sum()) for k in range(10)]
    axes[1].plot(mid / 1000, rate, "o-", color="#1f4e79")
    for m, r, n in zip(mid, rate, nbin):
        axes[1].text(m / 1000, r + 0.015, f"n={n}", ha="center", fontsize=8,
                     color="gray")
    axes[1].set_xlabel("Household income ($ thousands)")
    axes[1].set_ylabel("Share eligible for 401(k)")
    axes[1].set_title("Eligibility rises with income: the confounding")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def sensitivity_plot(rows, path):
    """rows: list of (label, estimate, ci_low, ci_high)."""
    labels = [r[0] for r in rows]
    est = np.array([r[1] for r in rows])
    lo = np.array([r[2] for r in rows])
    hi = np.array([r[3] for r in rows])
    ypos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.errorbar(est, ypos, xerr=[est - lo, hi - est], fmt="o", capsize=4,
                color="#1f4e79", ecolor="#1f4e79", markersize=6)
    ax.axvline(8000, color="#c0392b", linestyle="--", linewidth=1.5,
               label="True effect ($8,000)")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("DML estimate of the 401(k) effect ($)")
    ax.set_title("DML sensitivity: nuisance learner and number of folds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
