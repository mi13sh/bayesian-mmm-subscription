#!/usr/bin/env bash
# Rebuild everything from scratch: data, four main model fits, budget plans,
# charts, then four prior-sensitivity fits.
# Takes about 40 minutes on a 2-core codespace.
set -euo pipefail

rm -rf artifacts
python -m mmm.simulate
python scripts/make_lift_tests.py --data-dir data --seed 2026
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

# Prior sensitivity: ROI prior median 1.0 and 2.5 (1.5 is the main run above)
for roi in 1.0 2.5; do
  for cal in "" "_calibrated"; do
    dir="artifacts/sensitivity/roi_${roi}${cal}"
    python scripts/fit_model.py --target p_revenue --roi-prior "$roi" ${cal:+--calibrate} --out "$dir"
    python scripts/optimize_budget.py --target p_revenue --run-dir "$dir"
  done
done
python scripts/prior_sensitivity.py

echo "All done. Charts and tables are in artifacts/"
echo "For the robustness check across simulated worlds, run: bash scripts/run_robustness.sh"
