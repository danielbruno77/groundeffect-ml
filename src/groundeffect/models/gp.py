"""Gaussian-process surrogate and pool-based active learning.

The GP is the data-efficient baseline: with a few dozen CFD runs it already
gives calibrated uncertainty, and that uncertainty says where the next CFD run
is worth spending budget (wind tunnel and CFD hours are capped under the FIA
Aerodynamic Testing Restrictions).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from groundeffect.doe import DesignSpace


def make_gp(n_dims: int = 4, seed: int = 0) -> GaussianProcessRegressor:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(n_dims), length_scale_bounds=(1e-2, 1e2), nu=2.5) + WhiteKernel(
        1e-5, (1e-10, 1e-1)
    )
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3, random_state=seed)


@dataclass
class GPSurrogate:
    """One independent GP per scalar output, on unit-cube inputs."""

    space: DesignSpace = field(default_factory=DesignSpace)
    models: list[GaussianProcessRegressor] = field(default_factory=list)

    def fit(self, x: np.ndarray, y: np.ndarray, seed: int = 0) -> GPSurrogate:
        u = self.space.to_unit(x)
        self.models = [make_gp(u.shape[1], seed).fit(u, y[:, j]) for j in range(y.shape[1])]
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u = self.space.to_unit(x)
        out = [m.predict(u, return_std=True) for m in self.models]
        return np.column_stack([o[0] for o in out]), np.column_stack([o[1] for o in out])


def active_learning_curve(
    x_pool: np.ndarray,
    y_pool: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    n_init: int = 12,
    n_max: int = 96,
    step: int = 4,
    strategy: str = "max_std",
    seed: int = 0,
    target: int = 0,
) -> list[tuple[int, float]]:
    """Retrospective pool-based learning curve: (n_samples, test RMSE) pairs.

    ``strategy`` is ``max_std`` (pick the pool point with the largest predictive
    std), ``random`` or ``sequence`` (take the pool in its given order, e.g. the
    Sobol order, which is itself a space-filling design).
    """
    rng = np.random.default_rng(seed)
    space = DesignSpace()
    n = len(x_pool)
    chosen = list(range(n_init)) if strategy == "sequence" else list(rng.choice(n, n_init, replace=False))
    curve = []
    while True:
        gp = make_gp(x_pool.shape[1], seed).fit(space.to_unit(x_pool[chosen]), y_pool[chosen, target])
        pred = gp.predict(space.to_unit(x_test))
        curve.append((len(chosen), float(np.sqrt(np.mean((pred - y_test[:, target]) ** 2)))))
        if len(chosen) >= min(n_max, n):
            return curve
        rest = np.setdiff1d(np.arange(n), chosen)
        k = min(step, len(rest))
        if strategy == "max_std":
            # Greedy batch via the "kriging believer": pretend the GP mean is the
            # answer at each pick, so the next pick goes somewhere else.
            batch: list[int] = []
            believer = gp
            for _ in range(k):
                cand = np.setdiff1d(rest, batch)
                _, std = believer.predict(space.to_unit(x_pool[cand]), return_std=True)
                batch.append(int(cand[np.argmax(std)]))
                xs = space.to_unit(x_pool[chosen + batch])
                ys = np.r_[y_pool[chosen, target], gp.predict(space.to_unit(x_pool[batch]))]
                believer = GaussianProcessRegressor(kernel=gp.kernel_, optimizer=None, normalize_y=True).fit(xs, ys)
            chosen += batch
        elif strategy == "random":
            chosen += list(rng.choice(rest, k, replace=False))
        else:
            chosen += list(rest[:k])
