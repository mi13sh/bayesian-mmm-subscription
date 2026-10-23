"""Ground truth from the data generator, for checking the model's answers.

Only the evaluation code uses this. The model never sees it.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mmm.data import PAID, Prepared
from mmm.transforms import geometric_adstock, hill_saturation

TRUTH_FILE = {"p_revenue": "contributions_p_revenue.csv", "new_subs": "contributions_new_subs.csv"}


class TrueCurves:
    """The real steady-state response per paid channel.

    The generator's channel effect also varies week to week (price, launches,
    promos), so each curve is calibrated to the average multiplier over the
    same 52 weeks the optimiser treats as 'today'.
    """

    def __init__(self, df: pd.DataFrame, prep: Prepared, truth_dir="data/truth", weeks=52):
        params = json.loads((Path(truth_dir) / "params.json").read_text())["channels"]
        contrib = pd.read_csv(Path(truth_dir) / TRUTH_FILE[prep.target_name], index_col="week")
        window = slice(prep.n_train - weeks, prep.n_train)
        self.params, self.scale = params, {}
        for c in PAID:
            p = params[c]
            x = df[f"spend_{c}"].to_numpy() / 1000
            effect = hill_saturation(geometric_adstock(x, p["decay"]), p["half_sat"], p["slope"])
            self.scale[c] = contrib[c].iloc[window].sum() / effect[window].sum()

    def __call__(self, channel, weekly_spend):
        p = self.params[channel]
        x = np.asarray(weekly_spend, dtype=float) / 1000
        return self.scale[channel] * hill_saturation(x, p["half_sat"], p["slope"])

    def total(self, plan: dict) -> float:
        return sum(float(self(c, s)) for c, s in plan.items())

    def marginal(self, channel, weekly_spend) -> float:
        step = max(weekly_spend * 0.01, 1.0)
        return float((self(channel, weekly_spend + step) - self(channel, weekly_spend)) / step)
