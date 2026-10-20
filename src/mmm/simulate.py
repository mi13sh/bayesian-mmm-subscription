"""Generate a synthetic B2C subscription business with known ground truth.

The target is the predicted CLV (pCLV) of all new paid subscriptions in a week.
Paid media, email, product launches, promos and price changes all drive it.

Outputs:
    data/weekly.csv              what the model sees
    data/truth/params.json       the true parameters (never shown to the model)
    data/truth/contributions.csv true weekly pCLV contribution of each driver

Run:  python -m mmm.simulate
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mmm.transforms import geometric_adstock, hill_saturation

SEED = 2026
START = "2023-01-02"  # first Monday of 2023
N_WEEKS = 156

# --- True channel parameters --------------------------------------------------
# Spend-based inputs are in k EUR per week; email is in k accounts reached.
# max_subs: weekly new subscriptions the channel drives when fully saturated.
# clv: average predicted CLV (EUR) of a subscriber acquired through the channel,
#      at the base price, before plan-mix effects.
CHANNELS = {
    "search":    dict(decay=0.15, half_sat=9.0,   slope=2.0, max_subs=260, clv=190),
    "social":    dict(decay=0.40, half_sat=18.0,  slope=1.5, max_subs=380, clv=150),
    "display":   dict(decay=0.30, half_sat=4.0,   slope=1.2, max_subs=45,  clv=140),
    "affiliate": dict(decay=0.05, half_sat=5.0,   slope=1.5, max_subs=120, clv=110),
    "tv":        dict(decay=0.70, half_sat=8.0,   slope=1.5, max_subs=220, clv=200),
    "email":     dict(decay=0.20, half_sat=110.0, slope=1.5, max_subs=150, clv=230),
}
PAID = ["search", "social", "display", "affiliate", "tv"]

BASELINE = dict(subs=600, clv=210, growth_over_period=0.15)

# Seasonality: multiplicative bumps on the baseline, centred on a week of year.
SEASON_BUMPS = [
    dict(name="january_peak",   week=2,  width=2.5, size=0.22),
    dict(name="september_peak", week=37, width=2.5, size=0.16),
    dict(name="summer_lull",    week=31, width=4.0, size=-0.06),
    dict(name="december_dip",   week=51, width=1.5, size=-0.28),
]

# Product launches: permanent step changes from the launch week onwards.
# conv_lift multiplies all new subscriptions; clv_lift multiplies pCLV per sub.
PRODUCT_CHANGES = {
    "onboarding_v2":     dict(date="2023-09-04", conv_lift=0.06, clv_lift=0.00),
    "checkout_redesign": dict(date="2024-06-03", conv_lift=0.08, clv_lift=0.00),
    "annual_plan":       dict(date="2025-05-05", conv_lift=0.00, clv_lift=0.10),
}

# Monthly list price changes. Demand is inelastic: a higher price loses a few
# subscribers but each one is worth more.
PRICES = [("2023-01-02", 29.90), ("2024-04-01", 32.90), ("2025-03-03", 34.90)]
PRICE_ELASTICITY_SUBS = -0.7  # % change in subs per % change in price
PRICE_ELASTICITY_CLV = 0.9    # % change in pCLV per sub per % change in price

# Promo weeks (Black Friday and Cyber week): more subs, but discounted CLV.
PROMO_WEEKS = ["2023-11-20", "2023-11-27", "2024-11-25", "2024-12-02",
               "2025-11-24", "2025-12-01"]
PROMO = dict(conv_lift=0.25, clv_lift=-0.15)

# TV flights: (first week, number of weeks, weekly spend in k EUR)
TV_FLIGHTS = [("2023-02-27", 4, 15), ("2023-09-11", 3, 18), ("2024-01-15", 4, 12),
              ("2024-09-02", 4, 20), ("2025-02-10", 3, 10), ("2025-09-08", 4, 16)]

NOISE_SD = 0.05  # multiplicative noise on weekly subscriptions


def seasonality(dates: pd.DatetimeIndex) -> np.ndarray:
    """Multiplier around 1.0 built from the seasonal bumps."""
    woy = dates.isocalendar().week.to_numpy().astype(float)
    mult = np.ones(len(dates))
    for b in SEASON_BUMPS:
        # circular distance so week 52 and week 1 are neighbours
        d = np.minimum(np.abs(woy - b["week"]), 52 - np.abs(woy - b["week"]))
        mult += b["size"] * np.exp(-0.5 * (d / b["width"]) ** 2)
    return mult


def make_spend(dates: pd.DatetimeIndex, season: np.ndarray,
               rng: np.random.Generator) -> pd.DataFrame:
    """Weekly spend in EUR and email reach in accounts."""
    n = len(dates)
    demand = (season - season.mean()) / season.std()  # budgets follow demand

    search = 11.5 + 1.6 * demand + rng.normal(0, 0.8, n)
    # social: slow-moving campaign budget with occasional pushes
    social = 11 + np.cumsum(rng.normal(0, 0.9, n))
    social = social - np.linspace(0, social[-1] - 13, n)  # keep it from drifting off
    social += 3 * (rng.random(n) < 0.12)
    display = rng.uniform(2, 6, n)
    # affiliate partners scale with search demand, so the two move together
    affiliate = 0.45 * search + rng.normal(0, 0.5, n)

    tv = np.zeros(n)
    for start, weeks, level in TV_FLIGHTS:
        i = dates.get_loc(pd.Timestamp(start))
        tv[i:i + weeks] = level + rng.normal(0, 0.8, weeks)

    email = 105 + 35 * demand + rng.normal(0, 12, n)  # k accounts reached

    spend = pd.DataFrame({
        "spend_search": np.clip(search, 8, 15),
        "spend_social": np.clip(social, 5, 20),
        "spend_display": display,
        "spend_affiliate": np.clip(affiliate, 3, 8),
        "spend_tv": np.clip(tv, 0, 20),
    }, index=dates) * 1000
    spend["email_accounts_reached"] = np.clip(email, 60, 200) * 1000
    return spend.round(0)


def step_flags(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {f"flag_{k}": (dates >= pd.Timestamp(v["date"])).astype(int)
         for k, v in PRODUCT_CHANGES.items()},
        index=dates,
    )


def price_series(dates: pd.DatetimeIndex) -> pd.Series:
    price = pd.Series(np.nan, index=dates)
    for date, p in PRICES:
        price[dates >= pd.Timestamp(date)] = p
    return price


def simulate(seed: int = SEED):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START, periods=N_WEEKS, freq="W-MON")
    season = seasonality(dates)

    inputs = make_spend(dates, season, rng)
    flags = step_flags(dates)
    price = price_series(dates)
    promo = dates.isin(pd.to_datetime(PROMO_WEEKS)).astype(int)

    # Multipliers that act on every new subscriber, whatever the source
    price_ratio = price.to_numpy() / PRICES[0][1]
    conv_mult = price_ratio ** PRICE_ELASTICITY_SUBS * (1 + PROMO["conv_lift"] * promo)
    clv_mult = price_ratio ** PRICE_ELASTICITY_CLV * (1 + PROMO["clv_lift"] * promo)
    for k, v in PRODUCT_CHANGES.items():
        on = flags[f"flag_{k}"].to_numpy()
        conv_mult = conv_mult * (1 + v["conv_lift"] * on)
        clv_mult = clv_mult * (1 + v["clv_lift"] * on)

    # Expected subscriptions by source, before noise
    trend = 1 + BASELINE["growth_over_period"] * np.arange(N_WEEKS) / (N_WEEKS - 1)
    subs = {"baseline": BASELINE["subs"] * trend * season * conv_mult}
    clv = {"baseline": BASELINE["clv"] * clv_mult}

    col = {c: f"spend_{c}" for c in PAID} | {"email": "email_accounts_reached"}
    for name, p in CHANNELS.items():
        x = inputs[col[name]].to_numpy() / 1000
        effect = hill_saturation(geometric_adstock(x, p["decay"]), p["half_sat"], p["slope"])
        subs[name] = p["max_subs"] * effect * conv_mult
        clv[name] = p["clv"] * clv_mult

    contributions = pd.DataFrame({k: subs[k] * clv[k] for k in subs}, index=dates)
    expected_subs = sum(subs.values())
    expected_pclv = contributions.sum(axis=1).to_numpy()

    noise = rng.lognormal(0, NOISE_SD, N_WEEKS)
    new_subs = np.round(expected_subs * noise).astype(int)
    pclv = expected_pclv * noise

    weekly = inputs.copy()
    weekly["price_monthly_eur"] = price
    weekly["promo_week"] = promo
    weekly = weekly.join(flags)
    weekly["new_subs"] = new_subs
    weekly["pclv_new_subs_eur"] = pclv.round(0)
    weekly.index.name = "week"

    contributions.index.name = "week"
    truth = dict(seed=seed, channels=CHANNELS, baseline=BASELINE,
                 season_bumps=SEASON_BUMPS, product_changes=PRODUCT_CHANGES,
                 prices=PRICES, price_elasticity_subs=PRICE_ELASTICITY_SUBS,
                 price_elasticity_clv=PRICE_ELASTICITY_CLV, promo=PROMO,
                 noise_sd=NOISE_SD)
    return weekly, contributions, truth


def main(out_dir: str = "data") -> None:
    weekly, contributions, truth = simulate()
    out = Path(out_dir)
    (out / "truth").mkdir(parents=True, exist_ok=True)
    weekly.to_csv(out / "weekly.csv")
    contributions.round(0).to_csv(out / "truth" / "contributions.csv")
    (out / "truth" / "params.json").write_text(json.dumps(truth, indent=2))

    share = contributions.sum() / contributions.sum().sum()
    spend_cols = [c for c in weekly if c.startswith("spend_")]
    roi = {c: contributions[c].sum() / weekly[f"spend_{c}"].sum() for c in PAID}
    print(f"Saved {len(weekly)} weeks to {out / 'weekly.csv'}")
    print(f"Weekly pCLV: EUR {weekly['pclv_new_subs_eur'].min():,.0f} to "
          f"{weekly['pclv_new_subs_eur'].max():,.0f}")
    print(f"Weekly new subs: {weekly['new_subs'].min()} to {weekly['new_subs'].max()}")
    print(f"Paid media spend: EUR {weekly[spend_cols].sum(axis=1).mean():,.0f} per week")
    print("\nTrue share of pCLV:")
    print((share * 100).round(1).to_string())
    print("\nTrue pCLV return per EUR spent:")
    print(pd.Series(roi).round(2).to_string())


if __name__ == "__main__":
    main()
