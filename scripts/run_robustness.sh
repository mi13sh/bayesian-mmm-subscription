#!/usr/bin/env bash
# Robustness across simulated worlds: repeat the pRevenue analysis on four more
# simulated businesses (seeds 2027-2030), each with its own simulated lift tests,
# and count how often each plan beats today's split in truth.
# Run after run_all.sh: seed 2026 is the main world in artifacts/p_revenue[_calibrated].
# Data goes to data/seeds/, runs to artifacts/seeds/<seed>/ (both git-ignored);
# the summary is artifacts/seeds/runs.csv and summary.csv.
# Takes about 30 minutes on a 2-core codespace.
set -euo pipefail

rm -rf artifacts/seeds data/seeds
for seed in 2027 2028 2029 2030; do
  data="data/seeds/$seed"
  python -m mmm.simulate --seed "$seed" --out "$data"
  python scripts/make_lift_tests.py --data-dir "$data" --seed "$seed"
  for cal in "" "_calibrated"; do
    dir="artifacts/seeds/$seed/p_revenue$cal"
    python scripts/fit_model.py --target p_revenue ${cal:+--calibrate} --data-dir "$data" --out "$dir"
    python scripts/optimize_budget.py --target p_revenue --data-dir "$data" --run-dir "$dir"
  done
done
python scripts/seed_robustness.py

echo "Done. Summary in artifacts/seeds/summary.csv"
