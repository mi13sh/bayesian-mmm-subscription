"""Fit the MMM, check it, and compare its answers with the true ones.

Usage:
    python scripts/fit_model.py --target p_revenue
    python scripts/fit_model.py --target new_subs
    python scripts/fit_model.py --target p_revenue --calibrate
        adds the lift-test results in data/lift_tests.json to the model and
        writes to artifacts/p_revenue_calibrated/

Writes to artifacts/<target>/:
    prior_predictive.png   do the priors allow plausible data, before seeing any?
    fit.png                fitted vs actual, including the 12 held-out weeks
    recovery.png           estimated vs true return per EUR and share by channel
    recovery.csv           the same as a table
    diagnostics.json       convergence checks and holdout error
    posterior.nc           saved posterior, used by the app later
"""

import argparse
import json
import time
from pathlib import Path

import arviz as az
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from mmm.data import MEDIA, PAID, load_weekly, prepare
from mmm.model import build_model, roi_table

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
UNITS = {"p_revenue": "pRevenue (EUR / week)", "new_subs": "New subscriptions / week"}


def style(ax, title, dates=False):
    if dates:
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_title(title, loc="left", color=INK, fontsize=10)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)


def band(ax, dates, draws, color, label):
    lo, mid, hi = np.quantile(draws, [0.05, 0.5, 0.95], axis=0)
    ax.fill_between(dates, lo, hi, color=color, alpha=0.2, lw=0)
    ax.plot(dates, mid, color=color, lw=1.6, label=label)


def true_answers(prep, truth_dir="data/truth"):
    """True share and return per EUR over the training weeks."""
    col = {"p_revenue": "contributions_p_revenue.csv", "new_subs": "contributions_new_subs.csv"}
    contrib = pd.read_csv(Path(truth_dir) / col[prep.target_name], index_col="week")
    contrib = contrib.iloc[: prep.n_train]
    total = contrib.sum().sum()
    out = pd.DataFrame({"true_share": contrib[MEDIA].sum() / total})
    out["true_roi"] = pd.Series({c: contrib[c].sum() / prep.spend[c].iloc[: prep.n_train].sum()
                                 for c in PAID})
    return out


def main(target, draws, tune, chains, seed, calibrate=False):
    lift_tests = None
    out = Path("artifacts") / target
    if calibrate:
        lift_tests = json.loads(Path("data/lift_tests.json").read_text())[target]
        out = Path("artifacts") / f"{target}_calibrated"
        print(f"Calibrating with {len(lift_tests)} lift test(s)")
    out.mkdir(parents=True, exist_ok=True)
    df = load_weekly()
    prep = prepare(df, target)
    train, hold = slice(0, prep.n_train), slice(prep.n_train, len(df))
    scale = prep.y_scale

    # 1. Prior predictive check
    model = build_model(prep, train, lift_tests)
    with model:
        prior = pm.sample_prior_predictive(draws=500, random_seed=seed)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    band(ax, prep.dates[train], prior.prior_predictive["obs"].stack(s=("chain", "draw")).values.T * scale,
         MUTED, "Prior predictive (90% band)")
    ax.plot(prep.dates[train], prep.y[train] * scale, color=BLUE, lw=1.4, label="Actual")
    style(ax, f"Prior predictive check: {UNITS[target]}", dates=True)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "prior_predictive.png", dpi=150)
    plt.close(fig)

    # 2. Sample
    start = time.time()
    with model:
        idata = pm.sample(draws=draws, tune=tune, chains=chains, target_accept=0.95,
                          nuts_sampler="nutpie", random_seed=seed, progressbar=False)
    minutes = (time.time() - start) / 60

    # 3. Convergence
    summary = az.summary(idata, kind="diagnostics")
    divergences = int(idata.sample_stats["diverging"].sum())

    # 4. Posterior predictive on training weeks, then the held-out weeks
    with model:
        ppc_train = pm.sample_posterior_predictive(idata, var_names=["obs"], random_seed=seed)
        pm.set_data({"media_lags": prep.media_lags[hold], "controls": prep.controls[hold],
                     "y": np.zeros(hold.stop - hold.start)})
        ppc_hold = pm.sample_posterior_predictive(idata, var_names=["obs"], random_seed=seed,
                                                  predictions=True)
    pred_train = ppc_train.posterior_predictive["obs"].stack(s=("chain", "draw")).values.T * scale
    pred_hold = ppc_hold.predictions["obs"].stack(s=("chain", "draw")).values.T * scale
    actual = prep.y * scale

    def mape(a, p):
        return float(np.mean(np.abs(np.median(p, axis=0) - a) / a))

    lo, hi = np.quantile(pred_hold, [0.05, 0.95], axis=0)
    diagnostics = {
        "target": target,
        "sampling_minutes": round(minutes, 1),
        "divergences": divergences,
        "max_r_hat": float(summary["r_hat"].max()),
        "min_ess_bulk": float(summary["ess_bulk"].min()),
        "train_mape": round(mape(actual[train], pred_train), 4),
        "holdout_mape": round(mape(actual[hold], pred_hold), 4),
        "holdout_coverage_90": float(np.mean((actual[hold] >= lo) & (actual[hold] <= hi))),
    }
    (out / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))

    fig, ax = plt.subplots(figsize=(10, 3.5))
    band(ax, prep.dates[train], pred_train, BLUE, "Fitted (90% band)")
    band(ax, prep.dates[hold], pred_hold, ORANGE, "Forecast, held-out weeks")
    ax.plot(prep.dates, actual, color=INK, lw=1.0, label="Actual")
    ax.axvline(prep.dates[prep.n_train], color=MUTED, lw=0.8, ls="--")
    style(ax, f"Model fit: {UNITS[target]}  (holdout error {diagnostics['holdout_mape']:.1%})",
          dates=True)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "fit.png", dpi=150)
    plt.close(fig)

    # 5. Compare with the truth
    est = roi_table(idata, prep)
    recovery = est.join(true_answers(prep))
    recovery.round(4).to_csv(out / "recovery.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, metric, title, chans in [
        (axes[0], "roi", "Return per EUR of spend", PAID),
        (axes[1], "share", "Share of target", MEDIA),
    ]:
        r = recovery.loc[chans]
        ypos = np.arange(len(chans))[::-1]
        ax.errorbar(r[f"{metric}_mean"], ypos,
                    xerr=[r[f"{metric}_mean"] - r[f"{metric}_low"], r[f"{metric}_high"] - r[f"{metric}_mean"]],
                    fmt="o", color=BLUE, ms=6, capsize=3, lw=1.4, label="Estimate (90% interval)")
        ax.scatter(r[f"true_{metric}"], ypos, marker="D", s=40, color=ORANGE, zorder=3, label="Truth")
        ax.set_yticks(ypos, chans)
        if metric == "share":
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        style(ax, title)
        ax.grid(axis="x", color=GRID, lw=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center", ncol=2)
    fig.suptitle(f"Estimated vs true, target: {UNITS[target]}", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / "recovery.png", dpi=150)
    plt.close(fig)

    # 6. Save the posterior for the app
    idata.to_netcdf(str(out / "posterior.nc"))

    print(json.dumps(diagnostics, indent=2))
    print(recovery.round(3).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=["p_revenue", "new_subs"], default="p_revenue")
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--tune", type=int, default=1000)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--calibrate", action="store_true",
                   help="add the lift tests in data/lift_tests.json to the model")
    a = p.parse_args()
    main(a.target, a.draws, a.tune, a.chains, a.seed, a.calibrate)
