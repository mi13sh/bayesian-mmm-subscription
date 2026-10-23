"""Turn the weekly table into the scaled arrays the model needs.

Everything the model sees is scaled to roughly 0-1 so that priors can be set
in plain terms ("a channel drives at most ~30% of an average week") and the
same priors work for both targets.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

MEDIA = ["search", "social", "display", "affiliate", "tv", "email"]
MEDIA_COLUMNS = {c: f"spend_{c}" for c in MEDIA[:-1]} | {"email": "email_accounts_reached"}
PAID = MEDIA[:-1]
TARGETS = {"p_revenue": "p_revenue_eur", "new_subs": "new_subs"}

MAX_LAG = 12          # weeks of carryover the model can represent
FOURIER_ORDER = 6     # pairs of yearly sine/cosine terms
HOLDOUT_WEEKS = 12


@dataclass
class Prepared:
    dates: pd.DatetimeIndex
    target_name: str
    y: np.ndarray              # scaled target
    y_scale: float             # multiply by this to get back to EUR or subscribers
    media_lags: np.ndarray     # (weeks, lags, channels) scaled media, lag 0 first
    media_scale: np.ndarray    # per channel: raw units per 1.0 of scaled input
    spend: pd.DataFrame        # raw weekly spend in EUR, paid channels only
    controls: np.ndarray       # (weeks, controls)
    control_names: list
    n_train: int


def load_weekly(path: str = "data/weekly.csv") -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["week"], index_col="week")


def lag_matrix(x: np.ndarray, max_lag: int) -> np.ndarray:
    """(weeks, lags) array where column l holds x shifted back by l weeks."""
    out = np.zeros((len(x), max_lag + 1))
    for lag in range(max_lag + 1):
        out[lag:, lag] = x[: len(x) - lag]
    return out


def fourier_terms(dates: pd.DatetimeIndex, order: int) -> pd.DataFrame:
    day = dates.dayofyear.to_numpy()
    terms = {}
    for k in range(1, order + 1):
        angle = 2 * np.pi * k * day / 365.25
        terms[f"sin_{k}"] = np.sin(angle)
        terms[f"cos_{k}"] = np.cos(angle)
    return pd.DataFrame(terms, index=dates)


def prepare(df: pd.DataFrame, target: str = "p_revenue",
            holdout_weeks: int = HOLDOUT_WEEKS) -> Prepared:
    y_raw = df[TARGETS[target]].to_numpy(dtype=float)
    n_train = len(df) - holdout_weeks
    y_scale = y_raw[:n_train].mean()  # scale on training weeks only

    raw_media = np.column_stack([df[MEDIA_COLUMNS[c]].to_numpy(float) for c in MEDIA])
    media_scale = raw_media[:n_train].max(axis=0)
    scaled = raw_media / media_scale
    media_lags = np.stack([lag_matrix(scaled[:, i], MAX_LAG) for i in range(len(MEDIA))], axis=2)

    first_price = df["price_monthly_eur"].iloc[0]
    controls = pd.DataFrame({
        "trend": np.arange(len(df)) / (len(df) - 1),
        "promo_week": df["promo_week"],
        "flag_onboarding_v2": df["flag_onboarding_v2"],
        "flag_checkout_redesign": df["flag_checkout_redesign"],
        "flag_annual_plan": df["flag_annual_plan"],
        "log_price_ratio": np.log(df["price_monthly_eur"] / first_price),
    }, index=df.index).join(fourier_terms(df.index, FOURIER_ORDER))

    return Prepared(
        dates=df.index,
        target_name=target,
        y=y_raw / y_scale,
        y_scale=y_scale,
        media_lags=media_lags,
        media_scale=media_scale,
        spend=df[[MEDIA_COLUMNS[c] for c in PAID]].set_axis(PAID, axis=1),
        controls=controls.to_numpy(float),
        control_names=list(controls.columns),
        n_train=n_train,
    )
