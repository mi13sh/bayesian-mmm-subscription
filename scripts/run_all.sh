#!/usr/bin/env bash
# Rebuild everything from scratch: data, four model fits, budget plans, charts.
# Takes about 15-20 minutes on a 2-core codespace.
set -euo pipefail

rm -rf artifacts
python -m mmm.simulate
python scripts/plot_data.py
python -m pytest -q

for target in p_revenue new_subs; do
  python scripts/fit_model.py --target "$target"
  python scripts/fit_model.py --target "$target" --calibrate
  python scripts/optimize_budget.py --target "$target"
  python scripts/optimize_budget.py --target "$target" --calibrated
done
python scripts/optimize_budget.py --target p_revenue --budget-change 20
python scripts/optimize_budget.py --target p_revenue --calibrated --budget-change 20

echo "All done. Charts and tables are in artifacts/"
