import numpy as np

from mmm.simulate import simulate


def test_weekly_targets_match_subscriber_totals():
    weekly, subs, contrib_rev, contrib_subs, _ = simulate(seed=1)
    assert len(subs) == weekly["new_subs"].sum()
    assert np.allclose(contrib_subs.sum(axis=1), weekly["new_subs"])
    assert np.allclose(contrib_rev.sum(axis=1), weekly["p_revenue_eur"], rtol=1e-3)


def test_no_annual_plans_before_launch():
    _, subs, _, _, _ = simulate(seed=1)
    early = subs[subs["signup_week"] < "2025-05-05"]
    assert (early["plan"] == "monthly").all()


def test_prediction_model_overvalues_affiliate_more_than_search():
    _, subs, _, _, _ = simulate(seed=1)
    g = subs.groupby("source")
    bias = g["p_revenue_eur"].mean() / g["expected_revenue_24m"].mean()
    assert bias["affiliate"] > bias["search"]
