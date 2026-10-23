"""Response curves and budget allocation from a fitted MMM posterior.

Everything here works on a steady week: if a channel gets the same spend every
week, its adstocked input equals that spend (the adstock weights sum to 1), so
weekly contribution = beta * hill(spend / scale). That keeps the optimiser
simple and matches how budgets are actually set (a weekly or monthly run rate).

Two ways to choose a plan:
    risk = 0   maximise expected contribution. Tempting, but it acts on the
               posterior mean and ignores how unsure the model is.
    risk = 1   maximise expected gain minus one standard deviation of the gain
               (the default). A move only happens if it still pays off when
               the model is somewhat wrong. On this project's data, the risk-0
               plan loses money in the real world; the risk-1 plan does not.

TV is held at its current level by default. It runs in flights, and a steady
weekly TV spend is not a plan anyone would buy, so the optimiser should not
pretend to set it.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from mmm.data import MEDIA, PAID, Prepared


@dataclass
class Posterior:
    """Flattened posterior draws of the media parameters, one row per draw."""
    half_sat: np.ndarray   # (draws, channels)
    slope: np.ndarray
    beta: np.ndarray

    @classmethod
    def from_idata(cls, idata, max_draws: int | None = 1000, seed: int = 0):
        def flat(name):
            v = idata.posterior[name].stack(s=("chain", "draw")).transpose("s", ...).values
            return v
        half_sat, slope, beta = flat("half_sat"), flat("slope"), flat("beta")
        if max_draws and len(beta) > max_draws:
            idx = np.random.default_rng(seed).choice(len(beta), max_draws, replace=False)
            half_sat, slope, beta = half_sat[idx], slope[idx], beta[idx]
        return cls(half_sat, slope, beta)


def weekly_contribution(post: Posterior, prep: Prepared, spend: dict) -> np.ndarray:
    """Posterior draws of steady-state weekly contribution for a spend plan.

    `spend` maps paid channel -> weekly EUR. Returns (draws, len(spend)).
    """
    cols = [MEDIA.index(c) for c in spend]
    x = np.array(list(spend.values())) / prep.media_scale[cols]
    x = np.maximum(x, 1e-9)
    hs, sl, b = post.half_sat[:, cols], post.slope[:, cols], post.beta[:, cols]
    return b / (1 + (hs / x) ** sl) * prep.y_scale


def current_spend(prep: Prepared, weeks: int = 52) -> dict:
    """Average weekly spend per paid channel over the last `weeks` training weeks."""
    window = prep.spend.iloc[prep.n_train - weeks: prep.n_train]
    return window.mean().to_dict()


def response_curves(post: Posterior, prep: Prepared, max_multiple: float = 2.5,
                    points: int = 60) -> pd.DataFrame:
    """Weekly contribution and marginal return across a range of spend levels."""
    now = current_spend(prep)
    rows = []
    for c in PAID:
        grid = np.linspace(0, max_multiple * now[c], points)
        grid[0] = 1.0
        contrib = np.column_stack([weekly_contribution(post, prep, {c: s})[:, 0] for s in grid])
        step = max(now[c] * 0.01, 1.0)
        upper = np.column_stack([weekly_contribution(post, prep, {c: s + step})[:, 0] for s in grid])
        marginal = (upper - contrib) / step
        for i, s in enumerate(grid):
            rows.append({
                "channel": c, "weekly_spend": s,
                "contribution_mean": contrib[:, i].mean(),
                "contribution_low": np.quantile(contrib[:, i], 0.05),
                "contribution_high": np.quantile(contrib[:, i], 0.95),
                "marginal_mean": marginal[:, i].mean(),
                "marginal_low": np.quantile(marginal[:, i], 0.05),
                "marginal_high": np.quantile(marginal[:, i], 0.95),
            })
    return pd.DataFrame(rows)


def marginal_returns(post: Posterior, prep: Prepared, plan: dict | None = None) -> pd.DataFrame:
    """Extra target per extra EUR at a spend level (default: today's), with 90% intervals."""
    plan = plan or current_spend(prep)
    rows = {}
    for c, s in plan.items():
        step = max(s * 0.01, 1.0)
        m = (weekly_contribution(post, prep, {c: s + step}) - weekly_contribution(post, prep, {c: s}))[:, 0] / step
        rows[c] = {"weekly_spend": s, "marginal_mean": m.mean(),
                   "marginal_low": np.quantile(m, 0.05), "marginal_high": np.quantile(m, 0.95)}
    return pd.DataFrame(rows).T.rename_axis("channel")


def optimise(post: Posterior, prep: Prepared, total_budget: float | None = None,
             risk: float = 1.0, bounds: tuple = (0.5, 1.5), fixed: tuple = ("tv",)) -> pd.DataFrame:
    """Split a weekly budget across channels.

    total_budget  weekly EUR across all paid channels (default: today's total)
    risk          0 maximises expected contribution; higher values penalise
                  plans whose gain over today is uncertain (see module notes)
    bounds        each free channel may move between these multiples of today's
                  spend; a model should not be trusted far outside what it saw
    fixed         channels held at today's spend
    """
    now = current_spend(prep)
    total_budget = total_budget or sum(now.values())
    free = [c for c in PAID if c not in fixed]
    fixed_spend = {c: now[c] for c in fixed}
    free_budget = total_budget - sum(fixed_spend.values())
    scale = np.array([now[c] for c in free])
    lo, hi = bounds[0] * scale, bounds[1] * scale
    if free_budget < lo.sum() or free_budget > hi.sum():
        raise ValueError(
            f"Budget is outside what the bounds allow: free channels can take "
            f"EUR {lo.sum():,.0f} to {hi.sum():,.0f} per week, asked for {free_budget:,.0f}."
        )

    def plan(z):
        return dict(zip(free, z * scale)) | fixed_spend

    today = weekly_contribution(post, prep, {c: now[c] for c in PAID}).sum(axis=1)

    def objective(z):
        gain = weekly_contribution(post, prep, plan(z)).sum(axis=1) - today
        return -(gain.mean() - risk * gain.std()) / prep.y_scale

    # Several starting points, because the risk-adjusted objective is not convex
    ratio = free_budget / scale.sum()
    starts = [np.full(len(free), ratio)]
    rng = np.random.default_rng(0)
    for _ in range(5):
        z = rng.uniform(bounds[0], bounds[1], len(free))
        starts.append(z * free_budget / (z * scale).sum())
    best = None
    for z0 in starts:
        res = minimize(
            objective, np.clip(z0, *bounds), method="SLSQP",
            bounds=[bounds] * len(free),
            constraints=[{"type": "eq", "fun": lambda z: (z * scale).sum() / free_budget - 1}],
            options={"maxiter": 300, "ftol": 1e-10},
        )
        if res.success and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        raise RuntimeError("Optimiser did not converge from any starting point")
    chosen = plan(best.x)
    holds = np.isclose(total_budget, sum(now.values())) and \
        max(abs(chosen[c] / now[c] - 1) for c in free) < 0.005
    if holds:  # no move is worth the risk: keep today's split exactly
        chosen = dict(now)

    now_draws = weekly_contribution(post, prep, {c: now[c] for c in PAID})
    plan_draws = weekly_contribution(post, prep, {c: chosen[c] for c in PAID})
    out = pd.DataFrame({
        "current_spend": [now[c] for c in PAID],
        "planned_spend": [chosen[c] for c in PAID],
        "current_contribution": now_draws.mean(axis=0),
        "planned_contribution": plan_draws.mean(axis=0),
    }, index=pd.Index(PAID, name="channel"))
    out["change_%"] = 100 * (out["planned_spend"] / out["current_spend"] - 1)

    gain = plan_draws.sum(axis=1) - now_draws.sum(axis=1)
    out.attrs = {
        "risk": risk,
        "total_budget": total_budget,
        "current_total": float(now_draws.sum(axis=1).mean()),
        "planned_total": float(plan_draws.sum(axis=1).mean()),
        "gain_mean": float(gain.mean()),
        "gain_low": float(np.quantile(gain, 0.05)),
        "gain_high": float(np.quantile(gain, 0.95)),
        "prob_better": None if holds else float((gain > 0).mean()),
        "holds": bool(holds),
    }
    return out
