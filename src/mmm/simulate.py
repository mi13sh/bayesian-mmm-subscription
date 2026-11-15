"""Generate a synthetic multi-market B2C subscription business with known ground truth.

All money is in USD. Markets are grouped into three price tiers, and tier is
one of the subscriber attributes.

How the data comes about, week by week:
  1. Baseline demand, paid media and email drive how many people subscribe.
  2. Each new subscriber gets attributes: plan, market tier, device, payment
     method, age band, sale hook, promo discount and acquisition channel. The
     mix depends partly on what brought them in.
  3. Each subscriber's expected 24-month revenue follows from a churn process
     driven by those attributes. Acquisition channel has a small effect of its
     own, in line with what the business's CLV analysis has found.
  4. The business's pRevenue model predicts each subscriber's 24-month revenue
     from the same attributes. It is unbiased but imperfect for any single
     subscriber.
  5. The weekly MMM targets are the sum of pRevenue over the week's new
     subscribers, and the count of new subscribers.

Outputs:
    data/weekly.csv                          what the MMM sees
    data/subscribers.csv.gz                  one row per subscriber (for the CLV project)
    data/truth/params.json                   true parameters
    data/truth/contributions_p_revenue.csv   true weekly pRevenue by driver
    data/truth/contributions_new_subs.csv    true weekly new subscribers by driver
    data/truth/subscriber_truth.csv.gz       expected revenue and true source per subscriber

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
HORIZON_MONTHS = 24

# --- What drives subscriptions ---------------------------------------------------
# Spend is in k USD per week; email is in k accounts reached.
# max_subs: weekly new subscriptions the channel drives when fully saturated.
CHANNELS = {
    "search":     dict(decay=0.15, half_sat=19.0,  slope=2.0, max_subs=560),
    "social":     dict(decay=0.40, half_sat=38.0,  slope=1.5, max_subs=800),
    "display":    dict(decay=0.30, half_sat=7.0,   slope=1.2, max_subs=80),
    "influencer": dict(decay=0.35, half_sat=11.0,  slope=1.5, max_subs=260),
    "ctv":        dict(decay=0.60, half_sat=17.0,  slope=1.5, max_subs=460),
    "email":      dict(decay=0.20, half_sat=230.0, slope=1.5, max_subs=320),
}
PAID = ["search", "social", "display", "influencer", "ctv"]
SOURCES = ["baseline"] + list(CHANNELS)

BASELINE_SUBS = 1300
GROWTH_OVER_PERIOD = 0.15

# Seasonality: multiplicative bumps on baseline demand, centred on a week of year.
SEASON_BUMPS = [
    dict(name="january_peak",   week=2,  width=2.5, size=0.22),
    dict(name="september_peak", week=37, width=2.5, size=0.16),
    dict(name="summer_lull",    week=31, width=4.0, size=-0.06),
    dict(name="december_dip",   week=51, width=1.5, size=-0.28),
]

# Product launches: permanent from the launch week onwards.
# conv_lift multiplies all new subscriptions. The annual plan changes value
# through the plan mix, not through conversion.
PRODUCT_CHANGES = {
    "onboarding_v2":     dict(date="2023-09-04", conv_lift=0.06),
    "checkout_redesign": dict(date="2024-06-03", conv_lift=0.08),
    "annual_plan":       dict(date="2025-05-05", conv_lift=0.00),
}

# Tier 1 monthly list price. Tier 2 and 3 markets pay a share of it.
PRICES = [("2023-01-02", 19.99), ("2024-04-01", 21.99), ("2025-03-03", 23.99)]
TIER_PRICE = {"tier_1": 1.00, "tier_2": 0.75, "tier_3": 0.50}
PRICE_ELASTICITY_SUBS = -0.7
PRICE_EFFECT_ON_CHURN = 0.5  # churn hazard scales with (price / base price) ** this
ANNUAL_PRICE_MONTHS = 10     # annual plan costs 10x the monthly price

# Promo weeks (Black Friday and Cyber week): more subscribers, all discounted.
PROMO_WEEKS = ["2023-11-20", "2023-11-27", "2024-11-25", "2024-12-02",
               "2025-11-24", "2025-12-01"]
PROMO_CONV_LIFT = 0.25
DISCOUNT = 0.5  # first payment at half price

# CTV flights: (first week, number of weeks, weekly spend in k USD)
CTV_FLIGHTS = [("2023-02-06", 4, 32), ("2023-05-08", 3, 28), ("2023-09-04", 4, 38),
               ("2024-01-08", 4, 35), ("2024-04-15", 3, 30), ("2024-09-02", 4, 40),
               ("2025-01-13", 4, 33), ("2025-05-05", 4, 36), ("2025-09-08", 4, 38)]

WEEKLY_NOISE_SD = 0.04  # shared weekly shock on subscription volume

# --- What drives subscriber value ---------------------------------------------------
# Attribute mix: default shares, and the sources that shift them.
ATTRIBUTES = {
    "market_tier": dict(
        levels=["tier_1", "tier_2", "tier_3"], default=[0.55, 0.30, 0.15],
        by_source={"social": [0.50, 0.32, 0.18], "influencer": [0.45, 0.33, 0.22]}),
    "device": dict(
        levels=["desktop", "ios", "android"], default=[0.45, 0.33, 0.22],
        by_source={"social": [0.25, 0.42, 0.33], "influencer": [0.15, 0.50, 0.35],
                   "email": [0.60, 0.25, 0.15], "ctv": [0.50, 0.30, 0.20]}),
    "payment": dict(
        levels=["card", "paypal", "app_store", "wallet"], default=[0.45, 0.20, 0.20, 0.15],
        by_source={"social": [0.35, 0.15, 0.35, 0.15], "influencer": [0.30, 0.15, 0.40, 0.15]}),
    "age_band": dict(
        levels=["18-24", "25-34", "35-44", "45+"], default=[0.18, 0.36, 0.27, 0.19],
        by_source={"influencer": [0.35, 0.40, 0.17, 0.08], "ctv": [0.10, 0.30, 0.32, 0.28],
                   "social": [0.25, 0.38, 0.24, 0.13]}),
    "sale_hook": dict(
        levels=["free_trial", "feature_led", "bundle", "price_offer"],
        default=[0.35, 0.35, 0.15, 0.15],
        by_source={"search": [0.25, 0.50, 0.15, 0.10], "influencer": [0.45, 0.25, 0.10, 0.20]}),
}
ANNUAL_SHARE = dict(default=0.22, promo=0.32, by_source={"email": 0.28, "influencer": 0.16})
PROMO_HOOK_SHARE = 0.60  # in promo weeks most subscribers come in on the price offer

# Monthly churn hazard = base * exp(sum of effects). Positive means more churn.
BASE_MONTHLY_CHURN = 0.11
CHURN_EFFECTS = {
    "market_tier": {"tier_1": 0.0, "tier_2": 0.10, "tier_3": 0.25},
    "device": {"desktop": 0.0, "ios": -0.05, "android": 0.20},
    "payment": {"card": 0.0, "paypal": 0.10, "app_store": 0.05, "wallet": -0.05},
    "age_band": {"18-24": 0.30, "25-34": 0.05, "35-44": 0.0, "45+": -0.10},
    "sale_hook": {"free_trial": 0.20, "feature_led": -0.05, "bundle": -0.10, "price_offer": 0.15},
    "discounted": 0.25,
    # Acquisition channel on its own: small, as the business's CLV analysis found
    "acquisition_channel": {"organic": 0.0, "search": 0.0, "social": 0.03, "display": 0.02,
                            "influencer": 0.05, "ctv": 0.0, "email": -0.02},
}
ANNUAL_RENEWAL_RATE = 0.55  # at average churn risk

# The business's pRevenue model uses the same attributes. It is unbiased, but
# for any one subscriber it is off by this much (log-normal error).
PREDICTION_ERROR_SD = 0.20


def seasonality(dates: pd.DatetimeIndex) -> np.ndarray:
    """Multiplier around 1.0 built from the seasonal bumps."""
    woy = dates.isocalendar().week.to_numpy().astype(float)
    mult = np.ones(len(dates))
    for b in SEASON_BUMPS:
        d = np.minimum(np.abs(woy - b["week"]), 52 - np.abs(woy - b["week"]))
        mult += b["size"] * np.exp(-0.5 * (d / b["width"]) ** 2)
    return mult


def make_inputs(dates: pd.DatetimeIndex, season: np.ndarray,
                rng: np.random.Generator) -> pd.DataFrame:
    """Weekly spend in USD and email reach in accounts."""
    n = len(dates)
    demand = (season - season.mean()) / season.std()  # budgets follow demand

    search = 24 + 3.5 * demand + rng.normal(0, 1.6, n)
    social = 22 + np.cumsum(rng.normal(0, 1.8, n))     # slow-moving campaign budget
    social = social - np.linspace(0, social[-1] - 26, n)
    social += 6 * (rng.random(n) < 0.12)                # occasional pushes
    display = rng.uniform(4, 10, n)
    # influencer campaigns are booked around the same seasonal peaks as search
    influencer = 0.42 * search + rng.normal(0, 1.1, n)

    ctv = np.zeros(n)
    for start, weeks, level in CTV_FLIGHTS:
        i = dates.get_loc(pd.Timestamp(start))
        ctv[i:i + weeks] = level + rng.normal(0, 1.5, weeks)

    email = 220 + 70 * demand + rng.normal(0, 24, n)

    inputs = pd.DataFrame({
        "spend_search": np.clip(search, 17, 32),
        "spend_social": np.clip(social, 12, 34),
        "spend_display": display,
        "spend_influencer": np.clip(influencer, 6, 16),
        "spend_ctv": np.clip(ctv, 0, 42),
    }, index=dates) * 1000
    inputs["email_accounts_reached"] = np.clip(email, 120, 400) * 1000
    return inputs.round(0)


def weekly_drivers(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Price, promo and product-launch flags."""
    d = pd.DataFrame(index=dates)
    d["price_monthly_usd"] = np.nan
    for date, p in PRICES:
        d.loc[dates >= pd.Timestamp(date), "price_monthly_usd"] = p
    d["promo_week"] = dates.isin(pd.to_datetime(PROMO_WEEKS)).astype(int)
    for k, v in PRODUCT_CHANGES.items():
        d[f"flag_{k}"] = (dates >= pd.Timestamp(v["date"])).astype(int)
    return d


def expected_subs_by_source(inputs, drivers, season):
    """Expected weekly new subscriptions from each source, before noise."""
    price_ratio = drivers["price_monthly_usd"].to_numpy() / PRICES[0][1]
    conv = price_ratio ** PRICE_ELASTICITY_SUBS * (1 + PROMO_CONV_LIFT * drivers["promo_week"].to_numpy())
    for k, v in PRODUCT_CHANGES.items():
        conv = conv * (1 + v["conv_lift"] * drivers[f"flag_{k}"].to_numpy())

    trend = 1 + GROWTH_OVER_PERIOD * np.arange(N_WEEKS) / (N_WEEKS - 1)
    out = {"baseline": BASELINE_SUBS * trend * season * conv}
    col = {c: f"spend_{c}" for c in PAID} | {"email": "email_accounts_reached"}
    for name, p in CHANNELS.items():
        x = inputs[col[name]].to_numpy() / 1000
        effect = hill_saturation(geometric_adstock(x, p["decay"]), p["half_sat"], p["slope"])
        out[name] = p["max_subs"] * effect * conv
    return pd.DataFrame(out, index=inputs.index)


def draw_subscribers(expected, drivers, rng):
    """One row per subscriber with attributes and the source that brought them."""
    shock = rng.lognormal(0, WEEKLY_NOISE_SD, len(expected))
    counts = rng.poisson(expected.to_numpy() * shock[:, None])  # weeks x sources

    week_idx = np.repeat(np.arange(len(expected)), counts.sum(axis=1))
    source = np.concatenate([np.repeat(SOURCES, row) for row in counts])
    subs = pd.DataFrame({"signup_week": expected.index[week_idx], "source": source})
    subs["acquisition_channel"] = subs["source"].replace({"baseline": "organic"})

    for attr, spec in ATTRIBUTES.items():
        values = np.empty(len(subs), dtype=object)
        for src in SOURCES:
            m = (subs["source"] == src).to_numpy()
            probs = spec["by_source"].get(src, spec["default"])
            values[m] = rng.choice(spec["levels"], size=m.sum(), p=probs)
        subs[attr] = values

    d = drivers.iloc[week_idx].reset_index(drop=True)
    subs["discounted"] = d["promo_week"].to_numpy()
    promo_hook = (subs["discounted"] == 1) & (rng.random(len(subs)) < PROMO_HOOK_SHARE)
    subs.loc[promo_hook, "sale_hook"] = "price_offer"

    annual_p = subs["source"].map(ANNUAL_SHARE["by_source"]).fillna(ANNUAL_SHARE["default"]).to_numpy()
    annual_p = np.where(subs["discounted"] == 1, ANNUAL_SHARE["promo"], annual_p)
    annual_p = annual_p * d["flag_annual_plan"].to_numpy()  # no annual plan before launch
    subs["plan"] = np.where(rng.random(len(subs)) < annual_p, "annual", "monthly")
    subs["price_monthly_usd"] = (d["price_monthly_usd"].to_numpy()
                                 * subs["market_tier"].map(TIER_PRICE).to_numpy()).round(2)
    return subs


def churn_risk(subs: pd.DataFrame) -> np.ndarray:
    """Log multiplier on the monthly churn hazard for each subscriber."""
    risk = CHURN_EFFECTS["discounted"] * subs["discounted"].to_numpy(dtype=float)
    for attr in ["market_tier", "device", "payment", "age_band", "sale_hook", "acquisition_channel"]:
        risk = risk + subs[attr].map(CHURN_EFFECTS[attr]).to_numpy()
    base_price = PRICES[0][1] * subs["market_tier"].map(TIER_PRICE).to_numpy()
    return risk + PRICE_EFFECT_ON_CHURN * np.log(subs["price_monthly_usd"].to_numpy() / base_price)


def expected_revenue(subs: pd.DataFrame, risk: np.ndarray):
    """Expected revenue over the horizon, plus the churn and renewal rates behind it."""
    price = subs["price_monthly_usd"].to_numpy()
    first_payment_cut = np.where(subs["discounted"] == 1, DISCOUNT, 0.0)

    # Monthly plan: pays each month while still subscribed
    h = np.clip(BASE_MONTHLY_CHURN * np.exp(risk), 0.01, 0.9)
    expected_months = (1 - (1 - h) ** HORIZON_MONTHS) / h
    monthly = price * (expected_months - first_payment_cut)

    # Annual plan: pays up front, may renew once within the horizon
    renew = np.clip(ANNUAL_RENEWAL_RATE * np.exp(-0.5 * risk), 0, 0.95)
    annual = ANNUAL_PRICE_MONTHS * price * (1 - first_payment_cut + renew)

    return np.where(subs["plan"] == "annual", annual, monthly), h, renew


def simulate(seed: int = SEED):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START, periods=N_WEEKS, freq="W-MON")
    season = seasonality(dates)

    inputs = make_inputs(dates, season, rng)
    drivers = weekly_drivers(dates)
    expected = expected_subs_by_source(inputs, drivers, season)
    subs = draw_subscribers(expected, drivers, rng)

    exp_rev, h, renew = expected_revenue(subs, churn_risk(subs))
    subs["expected_revenue_24m"] = exp_rev

    # What the business's pRevenue model predicts: right on average, noisy per person
    err = rng.lognormal(-0.5 * PREDICTION_ERROR_SD**2, PREDICTION_ERROR_SD, len(subs))
    subs["p_revenue_usd"] = exp_rev * err

    # Realised revenue: one random draw of how long each subscriber stays
    price = subs["price_monthly_usd"].to_numpy()
    cut = np.where(subs["discounted"] == 1, DISCOUNT, 0.0)
    months = np.minimum(rng.geometric(h), HORIZON_MONTHS)
    renewed = rng.random(len(subs)) < renew
    subs["realised_revenue_24m"] = np.where(
        subs["plan"] == "annual",
        ANNUAL_PRICE_MONTHS * price * (1 - cut + renewed),
        price * (months - cut),
    )

    subs.insert(0, "subscriber_id", np.arange(1, len(subs) + 1))
    for c in ["expected_revenue_24m", "realised_revenue_24m", "p_revenue_usd"]:
        subs[c] = subs[c].round(2)

    # Weekly tables
    by_week = subs.groupby("signup_week")
    weekly = inputs.join(drivers)
    weekly["new_subs"] = by_week.size()
    weekly["p_revenue_usd"] = by_week["p_revenue_usd"].sum().round(0)
    weekly.index.name = "week"

    contrib_rev = subs.pivot_table(index="signup_week", columns="source",
                                   values="p_revenue_usd", aggfunc="sum", fill_value=0)
    contrib_subs = subs.pivot_table(index="signup_week", columns="source",
                                    values="subscriber_id", aggfunc="count", fill_value=0)
    contrib_rev = contrib_rev.reindex(columns=SOURCES, fill_value=0).round(0)
    contrib_subs = contrib_subs.reindex(columns=SOURCES, fill_value=0)
    for c in (contrib_rev, contrib_subs):
        c.index.name = "week"

    truth = dict(seed=seed, channels=CHANNELS, baseline_subs=BASELINE_SUBS,
                 growth_over_period=GROWTH_OVER_PERIOD, season_bumps=SEASON_BUMPS,
                 product_changes=PRODUCT_CHANGES, prices=PRICES, tier_price=TIER_PRICE,
                 price_elasticity_subs=PRICE_ELASTICITY_SUBS,
                 price_effect_on_churn=PRICE_EFFECT_ON_CHURN,
                 promo_conv_lift=PROMO_CONV_LIFT, discount=DISCOUNT,
                 ctv_flights=CTV_FLIGHTS, attributes=ATTRIBUTES, annual_share=ANNUAL_SHARE,
                 base_monthly_churn=BASE_MONTHLY_CHURN, churn_effects=CHURN_EFFECTS,
                 annual_renewal_rate=ANNUAL_RENEWAL_RATE,
                 prediction_error_sd=PREDICTION_ERROR_SD, weekly_noise_sd=WEEKLY_NOISE_SD)
    return weekly, subs, contrib_rev, contrib_subs, truth


def main(out_dir: str = "data") -> None:
    weekly, subs, contrib_rev, contrib_subs, truth = simulate()
    out = Path(out_dir)
    (out / "truth").mkdir(parents=True, exist_ok=True)
    for old in ["subscriber_source.csv.gz", "contributions.csv"]:
        (out / "truth" / old).unlink(missing_ok=True)

    weekly.to_csv(out / "weekly.csv")
    subs.drop(columns=["source", "expected_revenue_24m"]).to_csv(
        out / "subscribers.csv.gz", index=False)
    subs[["subscriber_id", "source", "expected_revenue_24m"]].to_csv(
        out / "truth" / "subscriber_truth.csv.gz", index=False)
    contrib_rev.to_csv(out / "truth" / "contributions_p_revenue.csv")
    contrib_subs.to_csv(out / "truth" / "contributions_new_subs.csv")
    (out / "truth" / "params.json").write_text(json.dumps(truth, indent=2))

    spend_cols = [f"spend_{c}" for c in PAID]
    print(f"Saved {len(weekly)} weeks and {len(subs):,} subscribers to {out}/")
    print(f"Weekly pRevenue: USD {weekly['p_revenue_usd'].min():,.0f} to "
          f"{weekly['p_revenue_usd'].max():,.0f}")
    print(f"Weekly new subs: {weekly['new_subs'].min():,} to {weekly['new_subs'].max():,}")
    total = weekly[spend_cols].sum(axis=1)
    print(f"Weekly media spend: USD {total.mean():,.0f} on average "
          f"(middle 80% of weeks {total.quantile(0.1):,.0f} to {total.quantile(0.9):,.0f})")

    spend = {c: weekly[f"spend_{c}"].sum() for c in PAID}
    g = subs.groupby("source")
    summary = pd.DataFrame({
        "share_subs_%": 100 * contrib_subs.sum() / contrib_subs.sum().sum(),
        "share_p_rev_%": 100 * contrib_rev.sum() / contrib_rev.sum().sum(),
        "p_rev_per_sub": g["p_revenue_usd"].mean(),
        "p_rev_per_usd": pd.Series({c: contrib_rev[c].sum() / spend[c] for c in PAID}),
    }).reindex(SOURCES)
    print("\nTrue picture by source:")
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()
