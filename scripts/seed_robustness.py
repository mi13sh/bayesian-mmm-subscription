"""Robustness across simulated worlds: how often does each plan beat today in truth?

Usage (after the runs below exist):
    python scripts/seed_robustness.py

Seed 2026 is the main world (artifacts/p_revenue[_calibrated]). The other
seeds are simulated, given their own lift tests, fitted and optimised by
scripts/run_robustness.sh, which ends by running this script.

Uses only budget_summary.json from each run (its true gains were written by the
evaluation code).

Writes to artifacts/seeds/:
    runs.csv      one row per seed, model and plan
    summary.csv   per model and plan: how often it beats today, and its typical true gain
"""

import json
from pathlib import Path

import pandas as pd

OUT = Path("artifacts/seeds")
SEEDS = [2026, 2027, 2028, 2029, 2030]
PLANS = {"expected_value": "Expected-value", "cautious": "Cautious"}


def run_dir(seed, calibrated):
    run = "p_revenue_calibrated" if calibrated else "p_revenue"
    return Path("artifacts") / run if seed == 2026 else OUT / str(seed) / run


def collect():
    rows = []
    for seed in SEEDS:
        for calibrated in [False, True]:
            summ = json.loads((run_dir(seed, calibrated) / "budget_summary.json").read_text())
            for key, label in PLANS.items():
                p = summ["plans"][key]
                rows.append({
                    "seed": seed, "lift_tests": calibrated, "plan": label,
                    "holds_today": p["holds_today"],
                    "prob_beats_today": p["prob_beats_today"],
                    "true_gain": round(p["true_gain_per_week"]),
                    "true_gain_%_of_media": p["true_gain_%_of_media"],
                })
    return pd.DataFrame(rows)


def summarise(runs):
    out = []
    for (cal, plan), g in runs.groupby(["lift_tests", "plan"], sort=False):
        out.append({
            "lift_tests": cal, "plan": plan, "worlds": len(g),
            "beats_today": int((g["true_gain"] > 0).sum()),
            "holds_today": int(g["holds_today"].sum()),
            "loses": int((g["true_gain"] < 0).sum()),
            "median_true_gain": round(g["true_gain"].median()),
            "worst_true_gain": int(g["true_gain"].min()),
            "best_true_gain": int(g["true_gain"].max()),
            "median_gain_%_of_media": round(g["true_gain_%_of_media"].median(), 1),
        })
    return pd.DataFrame(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = collect()
    summary = summarise(runs)
    runs.to_csv(OUT / "runs.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    print(runs.to_string(index=False), "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
