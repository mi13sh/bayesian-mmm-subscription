import importlib.util
from pathlib import Path

import numpy as np
import pymc as pm
import pytest

from mmm.data import load_weekly, prepare
from mmm.model import ROI_PRIOR, build_model

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fit_model_defaults_keep_main_runs():
    fit = load_script("fit_model")
    a = fit.parse_args(["--target", "p_revenue"])
    assert a.roi_prior is None and a.out is None
    assert fit.default_out("p_revenue") == Path("artifacts/p_revenue")
    assert fit.default_out("p_revenue", calibrate=True) == Path("artifacts/p_revenue_calibrated")


def test_fit_model_accepts_roi_prior_and_out():
    fit = load_script("fit_model")
    a = fit.parse_args(["--roi-prior", "2.5", "--calibrate", "--out", "artifacts/sensitivity/x"])
    assert a.roi_prior == 2.5 and a.calibrate
    assert a.out == Path("artifacts/sensitivity/x")
    with pytest.raises(SystemExit):
        fit.parse_args(["--roi-prior", "0"])


def test_optimize_budget_accepts_run_dir():
    opt = load_script("optimize_budget")
    assert opt.parse_args([]).run_dir is None
    a = opt.parse_args(["--run-dir", "artifacts/sensitivity/x"])
    assert a.run_dir == Path("artifacts/sensitivity/x")


def test_roi_median_moves_the_prior():
    prep = prepare(load_weekly(), "p_revenue")
    medians = {}
    for m in [None, 2.5]:
        with build_model(prep, roi_median=m):
            prior = pm.sample_prior_predictive(draws=2000, random_seed=1)
        medians[m] = float(np.median(prior.prior["roi"].values))
    assert medians[None] == pytest.approx(ROI_PRIOR["p_revenue"][0], rel=0.1)
    assert medians[2.5] == pytest.approx(2.5, rel=0.1)
    with pytest.raises(ValueError):
        build_model(prep, roi_median=-1.0)


def test_simulate_accepts_seed_and_out():
    from mmm.simulate import SEED, parse_args
    a = parse_args([])
    assert a.seed == SEED and a.out == "data"
    a = parse_args(["--seed", "2027", "--out", "data/seeds/2027"])
    assert a.seed == 2027 and a.out == "data/seeds/2027"


def test_scripts_accept_data_dir():
    fit, opt = load_script("fit_model"), load_script("optimize_budget")
    assert fit.parse_args([]).data_dir == Path("data")
    assert opt.parse_args([]).data_dir == Path("data")
    assert fit.parse_args(["--data-dir", "data/seeds/2027"]).data_dir == Path("data/seeds/2027")
    assert opt.parse_args(["--data-dir", "data/seeds/2027"]).data_dir == Path("data/seeds/2027")


def test_lift_tests_follow_the_recipe(tmp_path):
    """Same spends as the main tests, both targets, measured near the truth with one shared error."""
    import json
    import shutil

    for f in ["weekly.csv", "truth/params.json", "truth/contributions_p_revenue.csv",
              "truth/contributions_new_subs.csv"]:
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path("data") / f, tmp_path / f)
    mk = load_script("make_lift_tests")
    tests, true_lifts = mk.make_tests(tmp_path, seed=1)
    main = {t["channel"]: t for t in json.loads(Path("data/lift_tests.json").read_text())["p_revenue"]}
    for target in ["p_revenue", "new_subs"]:
        for t in tests[target]:
            m, true = main[t["channel"]], true_lifts[target][t["channel"]]
            assert t["spend_before"] == m["spend_before"] and t["spend_after"] == m["spend_after"]
            assert t["sd"] / abs(true) == pytest.approx(mk.REL_SE[t["channel"]], abs=0.02)
            assert abs(t["lift_per_week"] - true) < 4 * t["sd"]
    # one experiment: the error has the same sign for both targets
    for a, b in zip(tests["p_revenue"], tests["new_subs"]):
        ta, tb = true_lifts["p_revenue"][a["channel"]], true_lifts["new_subs"][b["channel"]]
        assert np.sign(a["lift_per_week"] - ta) == np.sign(b["lift_per_week"] - tb)
