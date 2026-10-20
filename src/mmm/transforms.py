"""Media transformations shared by the data generator and the model.

Both functions are plain NumPy so the generator and the tests can use them.
The PyMC model will use tensor versions of the same formulas.
"""

import numpy as np


def geometric_adstock(x: np.ndarray, decay: float, max_lag: int = 12) -> np.ndarray:
    """Carry part of each week's spend into the following weeks.

    Each week keeps `decay` of the previous week's effect. Weights are
    normalised to sum to 1, so the output stays in spend units and the
    total effect of a euro does not depend on the decay rate.
    """
    if not 0 <= decay < 1:
        raise ValueError("decay must be in [0, 1)")
    weights = decay ** np.arange(max_lag + 1)
    weights = weights / weights.sum()
    out = np.zeros(len(x), dtype=float)
    for lag, w in enumerate(weights[: len(x)]):
        out[lag:] += w * x[: len(x) - lag]
    return out


def hill_saturation(x: np.ndarray, half_sat: float, slope: float) -> np.ndarray:
    """Diminishing returns curve, from 0 (no spend) towards 1 (fully saturated).

    `half_sat` is the input level at which the channel reaches half its
    maximum effect. `slope` controls how sharply it bends: above 1 gives an
    S-shape, at or below 1 a concave curve.
    """
    x = np.asarray(x, dtype=float)
    return x**slope / (x**slope + half_sat**slope)
