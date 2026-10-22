"""Synthetic wind-tunnel campaign with known, injected faults.

Real tunnel data is confidential, so the detectors are developed against a
simulator that reproduces the failure modes that matter in practice, with
ground truth for every fault so detection can be scored:

* slow drift of the balance / dynamic-pressure calibration (temperature),
* step changes in the datum (balance re-zero, model re-fit, rolling-road belt change),
* single-sample spikes (electrical noise, debris),
* a blocked or leaking pressure tap (reads a near-constant value),
* configuration runs that disagree with the CFD/surrogate prediction beyond noise.

Each run is a ride-height sweep of a configuration. The programme interleaves
*datum runs* (a fixed reference configuration) between test runs, as tunnels
do to monitor repeatability.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

LoadModel = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]


def analytic_wing(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cheap ground-effect stand-in when no surrogate is supplied.

    Returns (CL_down, CD, Cp at 24 taps) for rows of (h/c, alpha, camber, thickness).
    The shape (downforce rising as 1/h then collapsing below h/c ~ 0.1) follows
    the classic single-element trend.
    """
    x = np.atleast_2d(x)
    h, a, m, _ = x.T
    base = 0.25 + 0.085 * a + 12.0 * m
    ge = 1.0 + 0.08 / (h + 0.04) - 0.9 * np.exp(-(((h - 0.0) / 0.07) ** 2))
    cl = base * ge
    cd = 0.012 + 0.03 * cl**2 / (np.pi * 4.0)
    xs = np.linspace(0.02, 0.98, 24)
    cp = -cl[:, None] * 1.6 * np.exp(-((xs[None, :] - 0.25) ** 2) / 0.08) + 0.2
    return cl, cd, cp


@dataclass
class Fault:
    kind: str  # drift | step | spike | tap_blocked | config_outlier
    start: int  # first affected sample index (global)
    end: int  # last affected sample index (inclusive)
    detail: str = ""


@dataclass
class Campaign:
    samples: pd.DataFrame  # one row per measurement point
    taps: np.ndarray  # [n_samples, n_taps] pressure-tap Cp
    faults: list[Fault] = field(default_factory=list)

    def truth(self, kind: str) -> np.ndarray:
        """Boolean mask over samples affected by faults of ``kind``."""
        mask = np.zeros(len(self.samples), dtype=bool)
        for f in self.faults:
            if f.kind == kind:
                mask[f.start : f.end + 1] = True
        return mask


DATUM = np.array([0.15, 4.0, 0.04, 0.12])


def simulate_campaign(
    load_model: LoadModel = analytic_wing,
    *,
    n_runs: int = 120,
    datum_every: int = 5,
    heights: tuple[float, ...] = (0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40),
    noise_cl: float = 0.004,
    noise_cd: float = 0.0004,
    noise_tap: float = 0.01,
    correlation: tuple[float, float] = (0.96, 0.01),
    seed: int = 7,
) -> Campaign:
    """Simulate a campaign. ``correlation`` maps model CL to tunnel CL (k, b):
    tunnels rarely match CFD exactly, and the detectors must learn that offset
    rather than flag it."""
    rng = np.random.default_rng(seed)
    rows, taps_all = [], []
    configs = []
    for run in range(n_runs):
        if run % datum_every == 0:
            configs.append(("datum", DATUM.copy()))
        else:
            cfg = np.array([np.nan, rng.uniform(1.0, 7.0), rng.uniform(0.025, 0.055), rng.uniform(0.10, 0.14)])
            configs.append(("test", cfg))

    t = 0
    for run, (kind, cfg) in enumerate(configs):
        x = np.tile(cfg, (len(heights), 1))
        x[:, 0] = heights
        cl_model, cd, cp = load_model(x)
        cl = correlation[0] * cl_model + correlation[1]
        for i, h in enumerate(heights):
            rows.append(
                {
                    "sample": t,
                    "run": run,
                    "run_type": kind,
                    "h_c": h,
                    "alpha_deg": cfg[1],
                    "camber": cfg[2],
                    "thickness": cfg[3],
                    "cl_model": cl_model[i],
                    "cl_true": cl[i],
                    "cd_true": cd[i],
                    "cl": cl[i] + rng.normal(0, noise_cl),
                    "cd": cd[i] + rng.normal(0, noise_cd),
                    "temperature_c": 22.0 + 0.002 * t + rng.normal(0, 0.05),
                }
            )
            taps_all.append(cp[i] + rng.normal(0, noise_tap, size=cp.shape[1]))
            t += 1
    df = pd.DataFrame(rows)
    taps = np.array(taps_all)
    n = len(df)
    faults: list[Fault] = []

    # 1. Slow calibration drift over a block of runs (multiplicative on CL).
    d0, d1 = int(0.30 * n), int(0.45 * n)
    ramp = np.linspace(0.0, 0.025, d1 - d0 + 1)
    df.loc[d0:d1, "cl"] *= 1.0 + ramp  # recalibrated after the block
    faults.append(Fault("drift", d0, d1, "dynamic-pressure calibration drift, up to +2.5% CL"))
    # 2. Balance re-zero error: a step offset from a given run onwards (until the next re-zero).
    s0, s1 = int(0.62 * n), int(0.75 * n)
    s0 = int(df.index[df["run"] == df.loc[s0, "run"]][0])
    s1 = int(df.index[df["run"] == df.loc[s1, "run"]][-1])
    df.loc[s0:s1, "cl"] += 0.03
    faults.append(Fault("step", s0, s1, "balance zero shifted by +0.03 CL"))
    # 3. Spikes.
    for idx in rng.choice(np.arange(10, n - 10), 6, replace=False):
        df.loc[idx, "cl"] += rng.choice([-1, 1]) * rng.uniform(0.06, 0.12)
        faults.append(Fault("spike", int(idx), int(idx), "single-sample spike"))
    # 4. A pressure tap blocks: reads a constant value with tiny noise.
    tap, b0 = 9, int(0.5 * n)
    b1 = b0 + 60
    taps[b0 : b1 + 1, tap] = taps[b0, tap] + rng.normal(0, 0.001, b1 - b0 + 1)
    faults.append(Fault("tap_blocked", b0, b1, f"tap {tap} blocked"))
    # 5. A test configuration that does not behave like the model (e.g. mis-set flap).
    test_runs = df.loc[(df["run_type"] == "test") & (df.index > int(0.85 * n)), "run"].unique()
    bad_run = int(test_runs[0])
    rows_bad = df.index[df["run"] == bad_run]
    df.loc[rows_bad, "cl"] *= 0.93
    faults.append(Fault("config_outlier", int(rows_bad[0]), int(rows_bad[-1]), f"run {bad_run} 7% low vs prediction"))

    return Campaign(samples=df, taps=taps, faults=faults)
