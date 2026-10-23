"""Response curves and budget plans from the fitted MMM, checked against the truth.

Usage:
    python scripts/optimize_budget.py --target p_revenue
    python scripts/optimize_budget.py --target p_revenue --calibrated
    python scripts/optimize_budget.py --target p_revenue --budget-change 20

Needs artifacts/<run>/posterior.nc from scripts/fit_model.py, where <run> is
the target name, plus "_calibrated" with --calibrated.

Compares two plans for the same weekly budget:
    expected value   maximises the model's expected contribution (risk 0)
    cautious         only moves budget where the gain survives the model's
                     uncertainty (risk 1, the default in mmm.optimize)

Writes to artifacts/<run>/:
    response_curves.png    weekly contribution vs spend per channel, with truth
    marginal_returns.png   extra target per extra EUR at today's spend, with truth
    budget_plan.png        today vs both plans
    budget_plan.csv        both plans as a table
    budget_summary.json    expected gain, its uncertainty, and the true gain
"""

import argparse
import json
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mmm.data import PAID, load_weekly, prepare
from mmm.optimize import Posterior, current_spend, marginal_returns, optimise, response_curves
from mmm.truth import TrueCurves

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
UNITS = {"p_revenue": ("pRevenue", "EUR"), "new_subs": ("New subscriptions", "subs")}
PLANS = {"expected_value": 0.0, "cautious": 1.0}


def style(ax, title):
    ax.set_title(title, loc="left", color=INK, fontsize=10)
    ax.grid(color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_response_curves(post, prep, truth, now, name, unit, path):
    curves = response_curves(post, prep)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, c in zip(axes.flat, PAID):
        r = curves[curves["channel"] == c]
        ax.fill_between(r["weekly_spend"], r["contribution_low"], r["contribution_high"],
                        color=BLUE, alpha=0.2, lw=0)
        ax.plot(r["weekly_spend"], r["contribution_mean"], color=BLUE, lw=1.8, label="Model (90% band)")
        ax.plot(r["weekly_spend"], truth(c, r["weekly_spend"]), color=ORANGE, lw=1.4, ls="--",
                label="Truth")
        ax.axvline(now[c], color=MUTED, lw=0.8, ls=":")
        ax.text(now[c], ax.get_ylim()[1], " today", color=MUTED, fontsize=8, va="top")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        style(ax, c)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, frameon=False, fontsize=9, loc="center left")
    axes.flat[-1].text(0, 0.25, f"x axis: steady weekly spend (EUR)\ny axis: weekly {name} ({unit})",
                       color=MUTED, fontsize=9, transform=axes.flat[-1].transAxes)
    fig.suptitle(f"Response curves: weekly {name} by steady weekly spend",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_marginal(marg, name, unit, path):
    fig, ax = plt.subplots(figsize=(8, 3.4))
    y = np.arange(len(marg))[::-1]
    ax.errorbar(marg["marginal_mean"], y,
                xerr=[marg["marginal_mean"] - marg["marginal_low"], marg["marginal_high"] - marg["marginal_mean"]],
                fmt="o", color=BLUE, ms=6, capsize=3, lw=1.4, label="Model (90% interval)")
    ax.scatter(marg["true_marginal"], y, marker="D", s=40, color=ORANGE, zorder=3, label="Truth")
    ax.set_yticks(y, marg.index)
    style(ax, f"Extra {name} per extra EUR at today's spend ({unit})")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_plans(plans, summary, unit, path):
    fig, ax = plt.subplots(figsize=(9, 4))
    y = np.arange(len(PAID))[::-1] * 1.2
    today = plans["expected_value"]["current_spend"]
    bars = [("Today", today, GRID),
            ("Expected-value plan", plans["expected_value"]["planned_spend"], ORANGE),
            ("Cautious plan", plans["cautious"]["planned_spend"], BLUE)]
    for k, (label, values, color) in enumerate(bars):
        ax.barh(y + 0.32 - 0.32 * k, values, height=0.3, color=color, label=label)
    for yi, c in zip(y, PAID):
        for k, key in enumerate(["expected_value", "cautious"], start=1):
            ch = plans[key].loc[c, "change_%"]
            txt = "held" if c == "tv" else f"{ch:+.0f}%"
            ax.text(plans[key].loc[c, "planned_spend"] * 1.02, yi + 0.32 - 0.32 * k, txt,
                    va="center", color=INK, fontsize=8)
    ax.set_yticks(y, PAID)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k"))
    ev, ca = summary["plans"]["expected_value"], summary["plans"]["cautious"]
    style(ax, f"Weekly spend (EUR). True gain per week: expected-value plan "
              f"{ev['true_gain_per_week']:+,.0f} {unit}, cautious plan {ca['true_gain_per_week']:+,.0f} {unit}")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(target, budget_change, calibrated):
    run = f"{target}_calibrated" if calibrated else target
    out = Path("artifacts") / run
    df = load_weekly()
    prep = prepare(df, target)
    post = Posterior.from_idata(az.from_netcdf(out / "posterior.nc"))
    truth = TrueCurves(df, prep)
    name, unit = UNITS[target]
    now = current_spend(prep)

    plot_response_curves(post, prep, truth, now, name, unit, out / "response_curves.png")

    marg = marginal_returns(post, prep)
    marg["true_marginal"] = [truth.marginal(c, now[c]) for c in marg.index]
    plot_marginal(marg, name, unit, out / "marginal_returns.png")

    total = sum(now.values()) * (1 + budget_change / 100)
    true_now = truth.total(now)
    plans, summary = {}, {"run": run, "budget_change_%": budget_change,
                          "weekly_budget_eur": round(total), "plans": {}}
    for key, risk in PLANS.items():
        plan = optimise(post, prep, total_budget=total, risk=risk)
        plans[key] = plan
        s = plan.attrs
        true_gain = truth.total(plan["planned_spend"].to_dict()) - true_now
        summary["plans"][key] = {
            "risk": risk,
            "changes_%": plan["change_%"].round(0).to_dict(),
            "expected_gain_per_week": round(s["gain_mean"], 2),
            "gain_90_interval": [round(s["gain_low"], 2), round(s["gain_high"], 2)],
            "holds_today": s["holds"],
            "prob_beats_today": None if s["holds"] else round(s["prob_better"], 3),
            "true_gain_per_week": round(true_gain, 2),
            "true_gain_%_of_media": round(100 * true_gain / true_now, 1),
        }
    summary["marginal_returns"] = marg.round(4).to_dict(orient="index")
    (out / "budget_summary.json").write_text(json.dumps(summary, indent=2))

    table = pd.concat({k: v[["planned_spend", "change_%"]] for k, v in plans.items()}, axis=1)
    table.insert(0, "current_spend", plans["expected_value"]["current_spend"])
    table.round(1).to_csv(out / "budget_plan.csv")
    plot_plans(plans, summary, unit, out / "budget_plan.png")

    print(f"Run: {run}, weekly budget EUR {total:,.0f}\n")
    print(marg.round(3).to_string(), "\n")
    print(table.round(0).to_string(), "\n")
    for key, v in summary["plans"].items():
        if v["holds_today"]:
            print(f"{key:15s} holds today's split: no move pays off once uncertainty is counted")
            continue
        print(f"{key:15s} expected {v['expected_gain_per_week']:+,.0f}  "
              f"P(beats today) {v['prob_beats_today']:.0%}  true {v['true_gain_per_week']:+,.0f} {unit}/week")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=["p_revenue", "new_subs"], default="p_revenue")
    p.add_argument("--budget-change", type=float, default=0.0,
                   help="percent change in total weekly budget, e.g. 20 or -10")
    p.add_argument("--calibrated", action="store_true",
                   help="use the posterior fitted with lift tests")
    a = p.parse_args()
    main(a.target, a.budget_change, a.calibrated)
