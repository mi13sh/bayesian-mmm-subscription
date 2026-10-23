"""Bayesian marketing mix model in PyMC.

Weekly target (scaled) =
      intercept
    + controls (trend, promo, product launches, price, yearly seasonality)
    + sum over channels of  beta * hill(adstock(media))
    + noise

Each channel has three parameters:
    decay      share of a week's effect that carries into the next week
    half_sat   scaled input at which the channel reaches half its maximum effect
    slope      how sharply returns diminish
and one effect size, beta: the most the channel can add in a week (scaled units).

The priors on paid channels are set on return per EUR (ROI), not on beta. A
prior like "paid channels usually return between 0.7 and 6 EUR of pRevenue per
EUR" is something a marketer can argue about; a prior on beta is not. Beta is
then whatever value delivers that ROI over the training weeks. This is the
approach Google's Meridian takes, and it stops the model from handing flat,
weakly identified channels a large effect that really belongs to the baseline.
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from mmm.data import MAX_LAG, MEDIA, PAID, Prepared

# Prior beliefs about carryover, as Beta(alpha, beta). TV is well known to carry
# over for weeks; the digital channels mostly act within a week or two.
DECAY_PRIOR = {c: (2.0, 3.0) for c in MEDIA} | {"tv": (4.0, 2.0)}

# How much each control is allowed to move a week, in scaled units (1.0 = an
# average week). Seasonality terms are kept small so they cannot absorb media.
CONTROL_PRIOR_SD = {
    "trend": 0.3, "promo_week": 0.3, "flag_onboarding_v2": 0.3,
    "flag_checkout_redesign": 0.3, "flag_annual_plan": 0.3, "log_price_ratio": 1.0,
}
FOURIER_PRIOR_SD = 0.05

# ROI prior per target: (median, spread on the log scale).
# pRevenue: median 2 EUR per EUR, 90% of prior mass between about 0.6 and 6.3.
# New subs: median 1 subscriber per 100 EUR (CPA of 100 EUR), same spread.
ROI_PRIOR = {"p_revenue": (2.0, 0.7), "new_subs": (0.01, 0.7)}
EMAIL_BETA_SD = 0.2  # email has no spend, so it keeps a prior on its effect size


def _media_effect(media, decay, half_sat, slope):
    """Adstock then hill, for a (weeks, lags, channels) tensor. Returns (weeks, channels)."""
    lags = np.arange(MAX_LAG + 1)[:, None]
    weights = decay[None, :] ** lags
    weights = weights / weights.sum(axis=0, keepdims=True)
    adstocked = (media * weights[None, :, :]).sum(axis=1)
    x = pt.maximum(adstocked, 1e-6)
    return 1 / (1 + (half_sat / x) ** slope)


def build_model(prep: Prepared, rows: slice | None = None,
                lift_tests: list | None = None) -> pm.Model:
    """Build the MMM on the given rows (training weeks by default).

    `lift_tests` is a list of experiment results, each a dict with channel,
    spend_before, spend_after (weekly EUR), lift_per_week and sd. Each one is
    added as an observation of how much the response curve rises between the
    two spend levels. A test that scales spend up pins down the slope of the
    curve where budget decisions are made, which the weekly data alone cannot.
    """
    rows = rows if rows is not None else slice(0, prep.n_train)
    coords = {"channel": MEDIA, "paid_channel": PAID, "lag": np.arange(MAX_LAG + 1), "control": prep.control_names}
    control_sd = np.array([CONTROL_PRIOR_SD.get(n, FOURIER_PRIOR_SD) for n in prep.control_names])
    decay_a = np.array([DECAY_PRIOR[c][0] for c in MEDIA])
    decay_b = np.array([DECAY_PRIOR[c][1] for c in MEDIA])

    with pm.Model(coords=coords) as model:
        media = pm.Data("media_lags", prep.media_lags[rows])
        controls = pm.Data("controls", prep.controls[rows])
        y = pm.Data("y", prep.y[rows])

        # --- Media ---
        decay = pm.Beta("decay", alpha=decay_a, beta=decay_b, dims="channel")
        half_sat = pm.LogNormal("half_sat", mu=np.log(0.6), sigma=0.4, dims="channel")
        slope = pm.Gamma("slope", alpha=6, beta=4, dims="channel")

        # Effect size implied by each paid channel's ROI over the training weeks
        roi_median, roi_sd = ROI_PRIOR[prep.target_name]
        roi = pm.LogNormal("roi", mu=np.log(roi_median), sigma=roi_sd, dims="paid_channel")
        train_media = pt.constant(prep.media_lags[: prep.n_train])
        train_sat = _media_effect(train_media, decay, half_sat, slope).sum(axis=0)
        n_paid = len(PAID)
        spend = prep.spend.iloc[: prep.n_train].sum().to_numpy()
        beta_paid = roi * spend / (prep.y_scale * train_sat[:n_paid])
        beta_email = pm.HalfNormal("beta_email", sigma=EMAIL_BETA_SD, shape=1)
        beta = pm.Deterministic("beta", pt.concatenate([beta_paid, beta_email]), dims="channel")

        saturated = _media_effect(media, decay, half_sat, slope)
        media_effect = (beta * saturated).sum(axis=1)

        # --- Baseline ---
        intercept = pm.Normal("intercept", mu=0.6, sigma=0.25)
        gamma = pm.Normal("gamma", mu=0, sigma=control_sd, dims="control")
        baseline = intercept + pt.dot(controls, gamma)

        # --- Lift tests ---
        for k, test in enumerate(lift_tests or []):
            i = MEDIA.index(test["channel"])
            before = max(test["spend_before"], 1.0) / prep.media_scale[i]
            after = test["spend_after"] / prep.media_scale[i]
            curve = lambda x: 1 / (1 + (half_sat[i] / x) ** slope[i])
            lift = beta[i] * (curve(after) - curve(before))
            pm.Normal(f"lift_test_{k}_{test['channel']}", mu=lift,
                      sigma=test["sd"] / prep.y_scale,
                      observed=test["lift_per_week"] / prep.y_scale)

        mu = baseline + media_effect
        sigma = pm.HalfNormal("sigma", sigma=0.1)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y, shape=mu.shape)

    return model


def _hill(x, half_sat, slope):
    x = np.maximum(x, 1e-6)
    return 1 / (1 + (half_sat / x) ** slope)


def contributions(idata, prep: Prepared, rows: slice | None = None,
                  media_lags: np.ndarray | None = None) -> np.ndarray:
    """Posterior weekly contribution of each channel, in target units.

    Returns an array (draws, weeks, channels). Pass `media_lags` to evaluate a
    what-if spend plan instead of the observed one.
    """
    rows = rows if rows is not None else slice(0, len(prep.dates))
    m = media_lags if media_lags is not None else prep.media_lags[rows]
    post = idata.posterior
    decay = post["decay"].stack(s=("chain", "draw")).transpose("s", ...).values
    half_sat = post["half_sat"].stack(s=("chain", "draw")).transpose("s", ...).values
    slope = post["slope"].stack(s=("chain", "draw")).transpose("s", ...).values
    beta = post["beta"].stack(s=("chain", "draw")).transpose("s", ...).values

    lags = np.arange(MAX_LAG + 1)[None, :, None]
    w = decay[:, None, :] ** lags
    w = w / w.sum(axis=1, keepdims=True)                         # (draws, lags, ch)
    adstocked = np.einsum("tlc,slc->stc", m, w)                  # (draws, weeks, ch)
    sat = _hill(adstocked, half_sat[:, None, :], slope[:, None, :])
    return beta[:, None, :] * sat * prep.y_scale


def roi_table(idata, prep: Prepared) -> pd.DataFrame:
    """Contribution per EUR of spend for each paid channel, over the training weeks."""
    rows = slice(0, prep.n_train)
    contrib = contributions(idata, prep, rows)                   # (draws, weeks, ch)
    total = contrib.sum(axis=1)                                  # (draws, ch)
    target_total = prep.y[rows].sum() * prep.y_scale
    out = []
    for i, c in enumerate(MEDIA):
        row = {"channel": c,
               "share_mean": total[:, i].mean() / target_total,
               "share_low": np.quantile(total[:, i], 0.05) / target_total,
               "share_high": np.quantile(total[:, i], 0.95) / target_total}
        if c in prep.spend:
            spend = prep.spend[c].iloc[rows].sum()
            row |= {"roi_mean": total[:, i].mean() / spend,
                    "roi_low": np.quantile(total[:, i], 0.05) / spend,
                    "roi_high": np.quantile(total[:, i], 0.95) / spend}
        out.append(row)
    return pd.DataFrame(out).set_index("channel")
