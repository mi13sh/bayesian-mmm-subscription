"""Quick check that PyMC installs, compiles and samples in this environment."""
import numpy as np
import pymc as pm
import arviz as az

print(f"PyMC {pm.__version__} | ArviZ {az.__version__} | NumPy {np.__version__}")

rng = np.random.default_rng(42)
x = rng.normal(size=200)
y = 1.5 + 2.0 * x + rng.normal(scale=0.5, size=200)

with pm.Model():
    a = pm.Normal("a", 0, 5)
    b = pm.Normal("b", 0, 5)
    sigma = pm.HalfNormal("sigma", 1)
    pm.Normal("y", a + b * x, sigma, observed=y)
    idata = pm.sample(draws=300, tune=300, chains=2, random_seed=42, progressbar=False)

print(az.summary(idata, var_names=["a", "b", "sigma"])[["mean", "sd", "r_hat"]])
print("\nSetup OK: expect a close to 1.5 and b close to 2.0")
