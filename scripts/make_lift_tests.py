"""Placeholder lift tests for another simulated world, made like the main ones.

Usage:
    python scripts/make_lift_tests.py --data-dir data/seeds/2027 --seed 2027

Writes <data-dir>/lift_tests.json with the same two tests as data/lift_tests.json:
    social      today's weekly spend raised by 50%
    influencer  today's weekly spend cut by 50%
"Today" is the average weekly spend over the last 52 training weeks, rounded to
10 USD, as in the main tests. The measured lift is the true lift (from that
world's truth/) plus a normal error whose standard deviation is the true lift
times the relative standard error of the matching main test (sd / lift in
data/lift_tests.json). Only the pRevenue target is written. The true lifts go
to <data-dir>/truth/lift_tests_truth.json, for evaluation only.

This stands in for running the experiment, so it reads the truth; the model
only ever sees the resulting lift_tests.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from mmm.data import load_weekly, prepare
from mmm.optimize import current_spend
from mmm.truth import TrueCurves

TESTS = {"social": 1.5, "influencer": 0.5}  # spend multiplier in the test regions
MAIN_TESTS = Path("data/lift_tests.json")


def relative_se(main_file=MAIN_TESTS, target="p_revenue"):
    tests = json.loads(Path(main_file).read_text())[target]
    return {t["channel"]: t["sd"] / abs(t["lift_per_week"]) for t in tests}


def make_tests(data_dir, seed, main_file=MAIN_TESTS):
    data_dir = Path(data_dir)
    df = load_weekly(data_dir / "weekly.csv")
    prep = prepare(df, "p_revenue")
    truth = TrueCurves(df, prep, truth_dir=data_dir / "truth")
    now = current_spend(prep)
    rel = relative_se(main_file)
    rng = np.random.default_rng(seed)
    tests, true_lifts = [], []
    for channel, mult in TESTS.items():
        before = int(round(now[channel], -1))
        after = round(before * mult)
        true_lift = float(truth(channel, after) - truth(channel, before))
        sd = rel[channel] * abs(true_lift)
        measured = true_lift + rng.normal(0, sd)
        true_lifts.append(true_lift)
        tests.append({
            "channel": channel, "spend_before": before, "spend_after": after,
            "lift_per_week": int(round(measured, -2)), "sd": int(round(sd, -2)),
            "source": f"placeholder {'scale-up' if mult > 1 else 'scale-down'} test, seed {seed}",
        })
    return tests, {t["channel"]: round(v) for t, v in zip(tests, true_lifts)}


def main(data_dir, seed):
    tests, true_lifts = make_tests(data_dir, seed)
    note = ("Placeholder lift tests for this simulated world, made by scripts/make_lift_tests.py "
            "like the main ones: true lift plus noise at the main tests' relative standard error.")
    Path(data_dir, "lift_tests.json").write_text(json.dumps({"_note": note, "p_revenue": tests}, indent=2))
    Path(data_dir, "truth", "lift_tests_truth.json").write_text(json.dumps(true_lifts, indent=2))
    for t in tests:
        print(f"{t['channel']:11s} {t['spend_before']:>8,.0f} -> {t['spend_after']:>8,.0f}  "
              f"measured {t['lift_per_week']:+,.0f} (sd {t['sd']:,.0f}), "
              f"true {true_lifts[t['channel']]:+,.0f}")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True, help="seed for the measurement error")
    return p.parse_args(argv)


if __name__ == "__main__":
    a = parse_args()
    main(a.data_dir, a.seed)
