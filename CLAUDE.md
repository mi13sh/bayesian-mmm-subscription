# Bayesian MMM for a B2C subscription business

Portfolio project by Milesh Vadodaria. Bayesian marketing mix model in PyMC on simulated data with a known ground truth, so every result can be checked against the truth.

## What the project shows
- A multi-market B2C subscription product (USD, three market price tiers), weekly data Jan 2023 to Dec 2025.
- Paid channels: search, social, display, influencer, ctv (bought in flights). Owned: email (accounts reached).
- Main target: pRevenue = predicted 24-month revenue of each week's new subscribers (`p_revenue_usd`). Comparison target: new subscriber count (`new_subs`).
- Headline: without experiments, a budget optimiser that trusts the model's central estimate loses money under every prior tested; lift tests fix it, and a cautious rule (expected gain minus one standard deviation) limits the damage.

## Layout
- `src/mmm/simulate.py` data generator and ground truth (writes `data/`)
- `src/mmm/transforms.py` adstock and Hill saturation
- `src/mmm/data.py` model inputs and scaling
- `src/mmm/model.py` PyMC model: ROI (return per dollar) priors on paid channels, optional lift tests as observations
- `src/mmm/optimize.py` response curves, marginal returns, risk-aware optimiser (ctv held fixed)
- `src/mmm/truth.py` true response curves, evaluation only
- `scripts/fit_model.py`, `scripts/optimize_budget.py`, `scripts/plot_data.py`, `scripts/make_lift_tests.py`, `scripts/seed_robustness.py`, `scripts/run_all.sh`, `scripts/run_robustness.sh`
- `data/lift_tests.json` simulated lift tests (true lift plus measurement error, `scripts/make_lift_tests.py`; true lifts in `data/truth/`). Project 2, a geo-test designer, covers running them
- `artifacts/<run>/` charts, tables, `posterior.nc` per run (`p_revenue`, `p_revenue_calibrated`, `new_subs`, `new_subs_calibrated`)

## Commands
- Tests: `python -m pytest -q`
- Fit: `python scripts/fit_model.py --target p_revenue [--calibrate]`
- Budget plans: `python scripts/optimize_budget.py --target p_revenue [--calibrated] [--budget-change 20]`
- Prior sensitivity: `python scripts/fit_model.py --target p_revenue --roi-prior 2.5 [--calibrate] --out artifacts/sensitivity/roi_2.5[_calibrated]`, then `python scripts/optimize_budget.py --target p_revenue --run-dir <that folder>`, then `python scripts/prior_sensitivity.py`
- Everything: `bash scripts/run_all.sh`, then `bash scripts/run_robustness.sh` (seeds 2027-2030; per-seed data and runs are git-ignored, only `artifacts/seeds/runs.csv` and `summary.csv` are committed)
- Sampler is nutpie (`nuts_sampler="nutpie"`), 4 chains x 1000 draws, target_accept 0.95.

## Rules
- The model and optimiser must never read `data/truth/`. Only `truth.py` and evaluation code may.
- Keep the fixed seed (2026) for the main results so README numbers stay reproducible.
- After any change to data or model, rerun the affected fits and update every number in README.md from the artifacts. Never write a number into the README that is not in an artifact.
- README style: concise, senior, plain language; explain technical terms in a few words; one or two lines of comment under each chart; no buzzwords (leveraged, spearheaded, unlock, seamless), minimal em dashes, no overclaiming.
- Charts: matplotlib, colours BLUE #2a78d6 (model), ORANGE #eb6834 (truth), INK #0b0b0b, MUTED #52514e, GRID #e4e3df; legends never covering data.

## Status
Done: generator, model, lift-test calibration, risk-aware optimiser, README.
To do before final:
1. [x] Prior sensitivity: refit with ROI prior median 1.0 and 2.5 (pRevenue) and report whether channel ranking and budget conclusions hold. Done: `artifacts/sensitivity/`, README section "Does the starting assumption matter?".
2. [x] Robustness across simulated worlds: rerun the pRevenue pipeline on 5 seeds and report how often each plan beats today in truth. Done: `scripts/run_robustness.sh`, `artifacts/seeds/`, README section "Does it hold beyond one simulated business?".
3. Headline chart at the top of the README.
4. Streamlit app and optional AI tab: deferred to the portfolio-wide apps phase.
