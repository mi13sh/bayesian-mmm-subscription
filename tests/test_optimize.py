import numpy as np
import pandas as pd
import pytest

from mmm.data import MEDIA, PAID, load_weekly, prepare
from mmm.optimize import Posterior, current_spend, optimise, weekly_contribution


@pytest.fixture(scope="module")
def setup():
    """A fake posterior: social pays far more per USD than display."""
    prep = prepare(load_weekly(), "p_revenue")
    rng = np.random.default_rng(0)
    n = 200
    half_sat = np.full((n, 6), 0.6) * rng.lognormal(0, 0.05, (n, 6))
    slope = np.full((n, 6), 1.5)
    beta = np.tile([0.1, 0.4, 0.02, 0.1, 0.1, 0.1], (n, 1))
    return Posterior(half_sat, slope, beta), prep


def test_contribution_grows_with_spend(setup):
    post, prep = setup
    low = weekly_contribution(post, prep, {"social": 5000})
    high = weekly_contribution(post, prep, {"social": 15000})
    assert (high > low).all()


def test_plan_spends_the_budget_and_respects_bounds(setup):
    post, prep = setup
    now = current_spend(prep)
    plan = optimise(post, prep, risk=0.0)
    assert plan["planned_spend"].sum() == pytest.approx(sum(now.values()), rel=1e-4)
    ratio = plan["planned_spend"] / plan["current_spend"]
    assert ratio.between(0.5 - 1e-6, 1.5 + 1e-6).all()
    assert plan.loc["ctv", "planned_spend"] == pytest.approx(now["ctv"])


def test_plan_moves_money_towards_the_better_channel(setup):
    post, prep = setup
    plan = optimise(post, prep, risk=0.0)
    assert plan.loc["social", "change_%"] > 0
    assert plan.loc["display", "change_%"] < 0


def test_true_best_plan_beats_today_and_respects_rules():
    from mmm.truth import TrueCurves

    df = load_weekly()
    prep = prepare(df, "p_revenue")
    truth = TrueCurves(df, prep)
    now = current_spend(prep)
    total = sum(now.values())
    best = truth.best_plan(now, total)
    assert sum(best.values()) == pytest.approx(total, rel=1e-6)
    assert best["ctv"] == pytest.approx(now["ctv"])
    for c in PAID:
        assert 0.5 * now[c] - 1e-6 <= best[c] <= 1.5 * now[c] + 1e-6
    assert truth.total(best) >= truth.total(now)
    # no random plan under the same rules does better
    rng = np.random.default_rng(1)
    free = [c for c in PAID if c != "ctv"]
    for _ in range(200):
        z = rng.uniform(0.5, 1.5, len(free))
        z *= (total - now["ctv"]) / sum(z[i] * now[c] for i, c in enumerate(free))
        if z.min() < 0.5 or z.max() > 1.5:
            continue
        other = {c: z[i] * now[c] for i, c in enumerate(free)} | {"ctv": now["ctv"]}
        assert truth.total(other) <= truth.total(best) + 1e-6
