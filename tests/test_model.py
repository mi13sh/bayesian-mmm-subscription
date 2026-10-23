import numpy as np
import pymc as pm

from mmm.data import MEDIA, load_weekly, prepare
from mmm.model import build_model


def test_model_builds_and_starts_finite():
    prep = prepare(load_weekly(), "p_revenue")
    model = build_model(prep)
    logp = model.compile_logp()(model.initial_point())
    assert np.isfinite(logp)


def test_prior_predictive_has_one_value_per_training_week():
    prep = prepare(load_weekly(), "new_subs")
    with build_model(prep):
        prior = pm.sample_prior_predictive(draws=20, random_seed=1)
    assert prior.prior_predictive["obs"].shape[-1] == prep.n_train
    assert prior.prior["beta"].shape[-1] == len(MEDIA)


def test_scaling_keeps_training_target_near_one():
    prep = prepare(load_weekly(), "p_revenue")
    assert abs(prep.y[: prep.n_train].mean() - 1) < 1e-9
    assert prep.media_lags.shape == (len(prep.dates), 13, len(MEDIA))
