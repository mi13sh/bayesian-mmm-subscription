"""Prior sensitivity: do the conclusions hold if the ROI prior median moves?

Usage (after the fits and budget plans below exist):
    python scripts/prior_sensitivity.py

Reads six pRevenue runs, ROI prior median 1.0, 1.5 and 2.5, each with and
without lift tests. 1.5 is the main result in artifacts/p_revenue[_calibrated];
the others come from
    python scripts/fit_model.py --target p_revenue --roi-prior 1.0 [--calibrate] \\
        --out artifacts/sensitivity/roi_1.0[_calibrated]
    python scripts/optimize_budget.py --target p_revenue \\
        --run-dir artifacts/sensitivity/roi_1.0[_calibrated]

Uses only recovery.csv and budget_summary.json from each run (the truth columns
there were written by the evaluation code).

Writes to artifacts/sensitivity/:
    summary.csv           one row per run
    prior_sensitivity.png marginal returns by run, and the true gain of each plan
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
OUT = Path("artifacts/sensitivity")
PRIORS = [1.0, 1.5, 2.5]


def run_dir(prior, calibrated):
    if prior == 1.5:
        return Path("artifacts") / ("p_revenue_calibrated" if calibrated else "p_revenue")
    return OUT / f"roi_{prior:.1f}{'_calibrated' if calibrated else ''}"


def ranking(values):
    return " > ".join(pd.Series(values).sort_values(ascending=False).index)


def summarise():
    rows, margs = [], {}
    for calibrated in [False, True]:
        for prior in PRIORS:
            d = run_dir(prior, calibrated)
            rec = pd.read_csv(d / "recovery.csv", index_col="channel").dropna(subset=["true_roi"])
            summ = json.loads((d / "budget_summary.json").read_text())
            marg = pd.DataFrame(summ["marginal_returns"]).T
            margs[(prior, calibrated)] = marg
            inside = ((rec["true_roi"] >= rec["roi_low"]) & (rec["true_roi"] <= rec["roi_high"]))
            ev, ca = summ["plans"]["expected_value"], summ["plans"]["cautious"]
            rows.append({
                "roi_prior": prior,
                "lift_tests": calibrated,
                "true_roi_inside_90": f"{inside.sum()}/{len(inside)}",
                "marginal_ranking": ranking(marg["marginal_mean"]),
                "social_marginal": round(marg.loc["social", "marginal_mean"], 2),
                "ev_social_change_%": ev["changes_%"]["social"],
                "ev_prob_beats_today": ev["prob_beats_today"],
                "ev_true_gain": round(ev["true_gain_per_week"]),
                "cautious_holds": ca["holds_today"],
                "cautious_true_gain": round(ca["true_gain_per_week"]),
            })
    table = pd.DataFrame(rows)
    true_rank = ranking(margs[(1.5, False)]["true_marginal"])
    return table, margs, true_rank


def plot(table, margs, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    channels = list(margs[(1.5, False)].index)
    y = np.arange(len(channels))[::-1]
    shades = {1.0: 0.35, 1.5: 0.65, 2.5: 1.0}
    offsets = {1.0: 0.22, 1.5: 0.0, 2.5: -0.22}
    for ax, calibrated, title in [(axes[0], False, "Without lift tests"),
                                  (axes[1], True, "With two lift tests")]:
        for prior in PRIORS:
            m = margs[(prior, calibrated)].loc[channels]
            ax.errorbar(m["marginal_mean"], y + offsets[prior],
                        xerr=[m["marginal_mean"] - m["marginal_low"], m["marginal_high"] - m["marginal_mean"]],
                        fmt="o", color=BLUE, alpha=shades[prior], ms=5, capsize=2, lw=1.2,
                        label=f"Model, prior median {prior:.1f}")
        ax.scatter(margs[(1.5, False)].loc[channels, "true_marginal"], y, marker="D", s=40,
                   color=ORANGE, zorder=3, label="Truth")
        ax.set_yticks(y, channels)
        ax.set_title(f"{title}: extra pRevenue per extra dollar", loc="left", color=INK, fontsize=10)
        ax.grid(axis="x", color=GRID, lw=0.6)

    ax = axes[2]
    labels, ev, ca = [], [], []
    for _, r in table.iterrows():
        labels.append(f"{r.roi_prior:.1f}, {'lift tests' if r.lift_tests else 'no tests'}")
        ev.append(r.ev_true_gain / 1000)
        ca.append(r.cautious_true_gain / 1000)
    yy = np.arange(len(labels))[::-1]
    ax.barh(yy + 0.18, ev, height=0.34, color=BLUE, label="Expected-value plan")
    ax.barh(yy - 0.18, ca, height=0.34, color=MUTED, label="Cautious plan")
    for yi, v in zip(yy, ca):
        if v == 0:
            ax.text(0.1, yi - 0.18, "holds", va="center", color=MUTED, fontsize=8)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(yy, labels)
    ax.set_title("True gain vs today (USD k / week), by prior median", loc="left", color=INK, fontsize=10)
    ax.grid(axis="x", color=GRID, lw=0.6)

    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(colors=MUTED, labelsize=8)
    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = axes[2].get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="lower center", ncol=6)
    fig.suptitle("Prior sensitivity: ROI prior median 1.0, 1.5 (main) and 2.5",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table, margs, true_rank = summarise()
    table.to_csv(OUT / "summary.csv", index=False)
    plot(table, margs, OUT / "prior_sensitivity.png")
    print(f"True marginal ranking: {true_rank}\n")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
