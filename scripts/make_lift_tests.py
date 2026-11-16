"""Simulated lift tests for a simulated world: true lift plus measurement error.

Usage:
    python scripts/make_lift_tests.py --data-dir data --seed 2026              (main world)
    python scripts/make_lift_tests.py --data-dir data/seeds/2027 --seed 2027

Writes <data-dir>/lift_tests.json with two 6-week geo tests:
    social      today's weekly spend raised by 50% in the test regions
    influencer  today's weekly spend cut by 50%
"Today" is the average weekly spend over the last 52 training weeks, rounded to
10 USD. The measured lift is the true lift (from that world's truth/) plus a
normal error whose standard deviation is a fixed share of the true lift
(REL_SE, about a quarter, typical of a geo test at this budget). Each test has
one error draw, shared by both targets, as one experiment would. The true lifts
go to <data-dir>/truth/lift_tests_truth.json, for evaluation only.

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
# Standard error as a share of the true lift (kept from the first placeholder readouts).
REL_SE = {"social": 3800 / 15000, "influencer": 1800 / 6600}
TARGETS = {"p_revenue": -2, "new_subs": 0}  # target -> rounding digits
NOTE = ("Simulated lift tests: two 6-week geo tests, paid social raised by 50% in test regions "
        "and influencer cut by 50%. Each result is the true lift plus realistic measurement error "
        "(standard error about a quarter of the lift), made by scripts/make_lift_tests.py with "
        "seed {seed}; the true lifts are in truth/lift_tests_truth.json, which the model never reads. "
        "Spend is weekly USD at total-business scale; lift_per_week is the measured change in the "
        "target per week (negative for a cut), sd its standard error.")


def make_tests(data_dir, seed):
    """Return ({target: [test, ...]}, {target: {channel: true lift}})."""
    data_dir = Path(data_dir)
    df = load_weekly(data_dir / "weekly.csv")
    preps = {t: prepare(df, t) for t in TARGETS}
    truths = {t: TrueCurves(df, p, truth_dir=data_dir / "truth") for t, p in preps.items()}
    now = current_spend(preps["p_revenue"])
    rng = np.random.default_rng(seed)
    tests = {t: [] for t in TARGETS}
    true_lifts = {t: {} for t in TARGETS}
    for channel, mult in TESTS.items():
        before = int(round(now[channel], -1))
        after = round(before * mult)
        z = rng.standard_normal()  # one experiment, one measurement error
        for target, digits in TARGETS.items():
            true_lift = float(truths[target](channel, after) - truths[target](channel, before))
            sd = REL_SE[channel] * abs(true_lift)
            measured = true_lift + z * sd
            true_lifts[target][channel] = round(true_lift, 1 if digits == 0 else 0)
            tests[target].append({
                "channel": channel, "spend_before": before, "spend_after": after,
                "lift_per_week": int(round(measured, digits)), "sd": int(round(sd, digits)),
                "source": f"simulated {'scale-up' if mult > 1 else 'scale-down'} geo test, seed {seed}",
            })
    return tests, true_lifts


def main(data_dir, seed):
    tests, true_lifts = make_tests(data_dir, seed)
    out = {"_note": NOTE.format(seed=seed), **tests}
    Path(data_dir, "lift_tests.json").write_text(json.dumps(out, indent=2) + "\n")
    Path(data_dir, "truth", "lift_tests_truth.json").write_text(json.dumps(true_lifts, indent=2) + "\n")
    for target, ts in tests.items():
        for t in ts:
            print(f"{target:9s} {t['channel']:11s} {t['spend_before']:>8,.0f} -> {t['spend_after']:>8,.0f}  "
                  f"measured {t['lift_per_week']:+,.0f} (sd {t['sd']:,.0f}), "
                  f"true {true_lifts[target][t['channel']]:+,.1f}")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True, help="seed for the measurement error")
    return p.parse_args(argv)


if __name__ == "__main__":
    a = parse_args()
    main(a.data_dir, a.seed)
