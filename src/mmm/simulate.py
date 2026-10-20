"""Generate a synthetic B2C subscription business with known ground truth.

How the data comes about, week by week:
  1. Baseline demand, paid media and email drive how many people subscribe.
  2. Each new subscriber gets attributes (plan, country, device, payment
     method, discount). The mix depends a little on what brought them in.
  3. Each subscriber's true 24-month revenue follows from a churn process
     driven by those attributes, plus a small effect of acquisition channel.
  4. A separate pRevenue model predicts each subscriber's 24-month revenue
     from attributes only, as a CLV model would in practice. It cannot see
     the channel, and it gets a recalibrated version in January 2025.
  5. The weekly MMM target is the sum of predicted revenue (pRevenue) of
     everyone who subscribed that week. New subscriptions are the second target.

Outputs:
    data/weekly.csv                    what the MMM sees
    data/subscribers.csv.gz            one row per subscriber (for the CLV project)
    data/truth/params.json             true parameters
    data/truth/contributions_p_revenue.csv  true weekly pRevenue by driver
    data/truth/contributions_new_subs.csv   true weekly new subs by driver
    data/truth/subscriber_source.csv.gz     what really brought each subscriber in

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

# --- What drives subscriptions ----------------------------------------------------
# Spend inputs are in k EUR per week; email is in k accounts reached.
# max_subs: weekly new subscriptions the channel drives when fully saturated.
CHANNELS = {
    "search":    dict(decay=0.15, half_sat=9.0,   slope=2.0, max_subs=260),
    "social":    dict(decay=0.40, half_sat=18.0,  slope=1.5, max_subs=380),
    "display":   dict(decay=0.30, half_sat=4.0,   slope=1.2, max_subs=45),
    "affiliate": dict(decay=0.05, half_sat=5.0,   slope=1.5, max_subs=120),
    "tv":        dict(decay=0.70, half_sat=8.0,   slope=1.5, max_subs=220),
    "email":     dict(decay=0.20, half_sat=110.0, slope=1.5, max_subs=150),
}
PAID = ["search", "social", "display", "affiliate", "tv"]
SOURCES = ["baseline"] + list(CHANNELS)

BASELINE_SUBS = 600
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
# through the plan mix below, not through conversion.
PRODUCT_CHANGES = {
    "onboarding_v2":     dict(date="2023-09-04", conv_lift=0.06),
    "checkout_redesign": dict(date="2024-06-03", conv_lift=0.08),
    "annual_plan":       dict(date="2025-05-05", conv_lift=0.00),
}

# Monthly list price. A higher price loses some subscribers and raises churn a
# little, but each payment is worth more.
PRICES = [("2023-01-02", 29.90), ("2024-04-01", 32.90), ("2025-03-03", 34.90)]
PRICE_ELASTICITY_SUBS = -0.7
PRICE_EFFECT_ON_CHURN = 0.5  # churn hazard scales with (price / base price) ** this
ANNUAL_PRICE_MONTHS = 10     # annual plan costs 10x the monthly price

# Promo weeks (Black Friday and Cyber week): more subscribers, all discounted.
PROMO_WEEKS = ["2023-11-20", "2023-11-27", "2024-11-25", "2024-12-02",
               "2025-11-24", "2025-12-01"]
PROMO_CONV_LIFT = 0.25
DISCOUNT = 0.5  # first payment at half price

# TV flights: (first week, number of weeks, weekly spend in k EUR)
TV_FLIGHTS = [("2023-02-27", 4, 15), ("2023-09-11", 3, 18), ("2024-01-15", 4, 12),
              ("2024-09-02", 4, 20), ("2025-02-10", 3, 10), ("2025-09-08", 4, 16)]

WEEKLY_NOISE_SD = 0.04  # shared weekly shock on subscription volume

# --- What drives subscriber value ---------------------------------------------------
# Attribute mix. Each source can shift the default mix slightly.
ATTRIBUTES = {
    "country": dict(levels=["DE", "AT_CH", "other"], default=[0.60, 0.20, 0.20],
                    by_source={"affiliate": [0.50, 0.20, 0.30]}),
    "device": dict(levels=["desktop", "mobile"], default=[0.55, 0.45],
                   by_source={"social": [0.30, 0.70], "display": [0.45, 0.55],
                              "email": [0.65, 0.35]}),
    "payment": dict(levels=["card", "paypal", "sepa"], default=[0.45, 0.35, 0.20],
                    by_source={"affiliate": [0.40, 0.45, 0.15]}),
}
ANNUAL_SHARE = dict(default=0.22, promo=0.32, by_source={"email": 0.28, "affiliate": 0.18})

# Monthly churn hazard = base * exp(sum of effects). Positive means more churn.
BASE_MONTHLY_CHURN = 0.12
CHURN_EFFECTS = {
    "country": {"DE": 0.0, "AT_CH": -0.10, "other": 0.25},
    "device": {"desktop": 0.0, "mobile": 0.20},
    "payment": {"card": 0.0, "paypal": 0.10, "sepa": -0.15},
    "discounted": 0.25,
}
# Small effect of channel beyond attributes, which the pRevenue model cannot see.
CHANNEL_CHURN_EFFECT = {"affiliate": 0.10, "social": 0.05}
ANNUAL_RENEWAL_RATE = 0.55  # at average churn risk

# The pRevenue model: what the business's CLV model believes. It ignores channel,
# slightly underestimates the mobile effect, and v2 (from Jan 2025) predicts 5%
# higher across the board after a recalibration.
PREDICTION_MODEL = dict(
    device_mobile_effect=0.15,
    v2_date="2025-01-06",
    v2_uplift=0.05,
)


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
    """Weekly spend in EUR and email reach in accounts."""
    n = len(dates)
    demand = (season - season.mean()) / season.std()  # budgets follow demand

    search = 11.5 + 1.6 * demand + rng.normal(0, 0.8, n)
    social = 11 + np.cumsum(rng.normal(0, 0.9, n))     # slow-moving campaign budget
    social = social - np.linspace(0, social[-1] - 13, n)
    social += 3 * (rng.random(n) < 0.12)                # occasional pushes
    display = rng.uniform(2, 6, n)
    affiliate = 0.45 * search + rng.normal(0, 0.5, n)   # moves with search demand

    tv = np.zeros(n)
    for start, weeks, level in TV_FLIGHTS:
        i = dates.get_loc(pd.Timestamp(start))
        tv[i:i + weeks] = level + rng.normal(0, 0.8, weeks)

    email = 105 + 35 * demand + rng.normal(0, 12, n)

    inputs = pd.DataFrame({
        "spend_search": np.clip(search, 8, 15),
        "spend_social": np.clip(social, 5, 20),
        "spend_display": display,
        "spend_affiliate": np.clip(affiliate, 3, 8),
        "spend_tv": np.clip(tv, 0, 20),
    }, index=dates) * 1000
    inputs["email_accounts_reached"] = np.clip(email, 60, 200) * 1000
    return inputs.round(0)


def weekly_drivers(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Price, promo and product-launch flags."""
    d = pd.DataFrame(index=dates)
    d["price_monthly_eur"] = np.nan
    for date, p in PRICES:
        d.loc[dates >= pd.Timestamp(date), "price_monthly_eur"] = p
    d["promo_week"] = dates.isin(pd.to_datetime(PROMO_WEEKS)).astype(int)
    for k, v in PRODUCT_CHANGES.items():
        d[f"flag_{k}"] = (dates >= pd.Timestamp(v["date"])).astype(int)
    return d


def expected_subs_by_source(inputs, drivers, season):
    """Expected weekly new subscriptions from each source, before noise."""
    price_ratio = drivers["price_monthly_eur"].to_numpy() / PRICES[0][1]
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

    for attr, spec in ATTRIBUTES.items():
        values = np.empty(len(subs), dtype=object)
        for src in SOURCES:
            m = (subs["source"] == src).to_numpy()
            probs = spec["by_source"].get(src, spec["default"])
            values[m] = rng.choice(spec["levels"], size=m.sum(), p=probs)
        subs[attr] = values

    d = drivers.iloc[week_idx].reset_index(drop=True)
    subs["discounted"] = d["promo_week"].to_numpy()
    annual_p = subs["source"].map(ANNUAL_SHARE["by_source"]).fillna(ANNUAL_SHARE["default"]).to_numpy()
    annual_p = np.where(subs["discounted"] == 1, ANNUAL_SHARE["promo"], annual_p)
    annual_p = annual_p * d["flag_annual_plan"].to_numpy()  # no annual plan before launch
    subs["plan"] = np.where(rng.random(len(subs)) < annual_p, "annual", "monthly")
    subs["price_monthly_eur"] = d["price_monthly_eur"].to_numpy()
    return subs


def churn_risk(subs, mobile_effect, include_channel):
    """Log multiplier on the churn hazard for each subscriber."""
    risk = (subs["country"].map(CHURN_EFFECTS["country"])
            + np.where(subs["device"] == "mobile", mobile_effect, 0.0)
            + subs["payment"].map(CHURN_EFFECTS["payment"])
            + CHURN_EFFECTS["discounted"] * subs["discounted"]
            + PRICE_EFFECT_ON_CHURN * np.log(subs["price_monthly_eur"] / PRICES[0][1]))
    if include_channel:
        risk = risk + subs["source"].map(CHANNEL_CHURN_EFFECT).fillna(0.0)
    return risk.to_numpy()


def expected_revenue(subs, risk):
    """Expected revenue over the horizon given a churn risk multiplier."""
    price = subs["price_monthly_eur"].to_numpy()
    first_payment_cut = np.where(subs["discounted"] == 1, DISCOUNT, 0.0)

    # Monthly plan: pays each month while still subscribed
    h = np.clip(BASE_MONTHLY_CHURN * np.exp(risk), 0.01, 0.9)
    expected_months = (1 - (1 - h) ** HORIZON_MONTHS) / h
    monthly = price * (expected_months - first_payment_cut)

    # Annual plan: pays up front, may renew once within the horizon
    annual_price = ANNUAL_PRICE_MONTHS * price
    renew = np.clip(ANNUAL_RENEWAL_RATE * np.exp(-0.5 * risk), 0, 0.95)
    annual = annual_price * (1 - first_payment_cut + renew)

    return np.where(subs["plan"] == "annual", annual, monthly), h, renew


def simulate(seed: int = SEED):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START, periods=N_WEEKS, freq="W-MON")
    season = seasonality(dates)

    inputs = make_inputs(dates, season, rng)
    drivers = weekly_drivers(dates)
    expected = expected_subs_by_source(inputs, drivers, season)
    subs = draw_subscribers(expected, drivers, rng)

    # True value: attributes plus the hidden channel effect
    true_risk = churn_risk(subs, CHURN_EFFECTS["device"]["mobile"], include_channel=True)
    subs["expected_revenue_24m"], h, renew = expected_revenue(subs, true_risk)

    # Realised revenue: one random draw of how long each subscriber stays
    price = subs["price_monthly_eur"].to_numpy()
    cut = np.where(subs["discounted"] == 1, DISCOUNT, 0.0)
    months = np.minimum(rng.geometric(h), HORIZON_MONTHS)
    renewed = rng.random(len(subs)) < renew
    subs["realised_revenue_24m"] = np.where(
        subs["plan"] == "annual",
        ANNUAL_PRICE_MONTHS * price * (1 - cut + renewed),
        price * (months - cut),
    )

    # What the business's pRevenue model predicts
    pred_risk = churn_risk(subs, PREDICTION_MODEL["device_mobile_effect"], include_channel=False)
    p_rev, _, _ = expected_revenue(subs, pred_risk)
    v2 = (subs["signup_week"] >= pd.Timestamp(PREDICTION_MODEL["v2_date"])).to_numpy()
    subs["p_revenue_eur"] = p_rev * np.where(v2, 1 + PREDICTION_MODEL["v2_uplift"], 1.0)
    subs["prediction_model_version"] = np.where(v2, "v2", "v1")

    subs.insert(0, "subscriber_id", np.arange(1, len(subs) + 1))
    for c in ["expected_revenue_24m", "realised_revenue_24m", "p_revenue_eur"]:
        subs[c] = subs[c].round(2)

    # Weekly tables
    by_week = subs.groupby("signup_week")
    weekly = inputs.join(drivers)
    weekly["new_subs"] = by_week.size()
    weekly["p_revenue_eur"] = by_week["p_revenue_eur"].sum().round(0)
    weekly.index.name = "week"

    contrib_rev = subs.pivot_table(index="signup_week", columns="source",
                                   values="p_revenue_eur", aggfunc="sum", fill_value=0)
    contrib_subs = subs.pivot_table(index="signup_week", columns="source",
                                    values="subscriber_id", aggfunc="count", fill_value=0)
    contrib_rev = contrib_rev.reindex(columns=SOURCES, fill_value=0).round(0)
    contrib_subs = contrib_subs.reindex(columns=SOURCES, fill_value=0)
    for c in (contrib_rev, contrib_subs):
        c.index.name = "week"

    truth = dict(seed=seed, channels=CHANNELS, baseline_subs=BASELINE_SUBS,
                 growth_over_period=GROWTH_OVER_PERIOD, season_bumps=SEASON_BUMPS,
                 product_changes=PRODUCT_CHANGES, prices=PRICES,
                 price_elasticity_subs=PRICE_ELASTICITY_SUBS,
                 price_effect_on_churn=PRICE_EFFECT_ON_CHURN,
                 promo_conv_lift=PROMO_CONV_LIFT, discount=DISCOUNT,
                 attributes=ATTRIBUTES, annual_share=ANNUAL_SHARE,
                 base_monthly_churn=BASE_MONTHLY_CHURN, churn_effects=CHURN_EFFECTS,
                 channel_churn_effect=CHANNEL_CHURN_EFFECT,
                 annual_renewal_rate=ANNUAL_RENEWAL_RATE,
                 prediction_model=PREDICTION_MODEL, weekly_noise_sd=WEEKLY_NOISE_SD)
    return weekly, subs, contrib_rev, contrib_subs, truth


def main(out_dir: str = "data") -> None:
    weekly, subs, contrib_rev, contrib_subs, truth = simulate()
    out = Path(out_dir)
    (out / "truth").mkdir(parents=True, exist_ok=True)

    weekly.to_csv(out / "weekly.csv")
    subs.drop(columns=["source", "expected_revenue_24m"]).to_csv(
        out / "subscribers.csv.gz", index=False)
    subs[["subscriber_id", "source", "expected_revenue_24m"]].to_csv(
        out / "truth" / "subscriber_source.csv.gz", index=False)
    contrib_rev.to_csv(out / "truth" / "contributions_p_revenue.csv")
    contrib_subs.to_csv(out / "truth" / "contributions_new_subs.csv")
    (out / "truth" / "params.json").write_text(json.dumps(truth, indent=2))

    print(f"Saved {len(weekly)} weeks and {len(subs):,} subscribers to {out}/")
    print(f"Weekly pRevenue: EUR {weekly['p_revenue_eur'].min():,.0f} to "
          f"{weekly['p_revenue_eur'].max():,.0f}")
    print(f"Weekly new subs: {weekly['new_subs'].min()} to {weekly['new_subs'].max()}")

    spend = {c: weekly[f"spend_{c}"].sum() for c in PAID}
    g = subs.groupby("source")
    summary = pd.DataFrame({
        "share_subs_%": 100 * contrib_subs.sum() / contrib_subs.sum().sum(),
        "share_p_rev_%": 100 * contrib_rev.sum() / contrib_rev.sum().sum(),
        "p_rev_per_sub": g["p_revenue_eur"].mean(),
        "true_rev_per_sub": g["expected_revenue_24m"].mean(),
        "p_rev_per_eur": pd.Series({c: contrib_rev[c].sum() / spend[c] for c in PAID}),
    }).reindex(SOURCES)
    print("\nTrue picture by source:")
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()
