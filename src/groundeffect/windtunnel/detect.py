"""Wind-tunnel data quality: drift, step, spike, tap and configuration checks.

The core idea is to monitor *residuals*, not raw loads: measured CL minus the
CFD/surrogate prediction mapped through a CFD-to-tunnel correlation learned on
a trusted baseline period. Configuration changes then stop looking like
anomalies, and what remains is instrument and process behaviour.

Detectors:
* Hampel filter on sample residuals            -> spikes
* EWMA + two-sided CUSUM on datum-run means    -> drift and steps (repeatability)
* PELT change-point segmentation on run means  -> when a level change happened
* robust per-run residual test                 -> single configurations that disagree
* PCA reconstruction error per pressure tap    -> blocked/leaking taps
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

MAD_TO_SIGMA = 1.4826


def robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(MAD_TO_SIGMA * np.median(np.abs(x - np.median(x)))) or float(np.std(x)) or 1e-12


def hampel(x: np.ndarray, window: int = 7, n_sigma: float = 4.0, sigma: float | None = None) -> np.ndarray:
    """Flag points further than ``n_sigma`` robust sigmas from the rolling median."""
    s = pd.Series(x)
    med = s.rolling(2 * window + 1, center=True, min_periods=3).median()
    sig = sigma if sigma is not None else MAD_TO_SIGMA * (s - med).abs().rolling(2 * window + 1, center=True, min_periods=3).median()
    return ((s - med).abs() > n_sigma * sig).to_numpy()


def ewma(x: np.ndarray, lam: float = 0.3) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    acc = 0.0
    for i, v in enumerate(x):
        acc = lam * v + (1 - lam) * acc  # z_0 = target (0): matches ewma_limits
        out[i] = acc
    return out


def ewma_limits(sigma: float, lam: float, n: int, L: float = 3.0) -> np.ndarray:
    i = np.arange(1, n + 1)
    return L * sigma * np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * i)))


def cusum(x: np.ndarray, sigma: float, k: float = 0.5, h: float = 5.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-sided tabular CUSUM on standardised data. Returns (S+, S-, alarm)."""
    z = np.asarray(x) / sigma
    sp, sm = np.zeros_like(z), np.zeros_like(z)
    for i in range(len(z)):
        prev_p, prev_m = (sp[i - 1], sm[i - 1]) if i else (0.0, 0.0)
        sp[i] = max(0.0, prev_p + z[i] - k)
        sm[i] = max(0.0, prev_m - z[i] - k)
    return sp, sm, (sp > h) | (sm > h)


def datum_chart(
    dev: np.ndarray, sigma: float, lam: float = 0.3, k: float = 0.5, h: float = 4.0, shewhart: float = 3.0
) -> dict[str, np.ndarray]:
    """Combined Shewhart / EWMA / CUSUM chart on datum-run deviations.

    After an alarm the statistics restart (the tunnel is investigated and
    re-zeroed), so a fault that has been cleared does not keep alarming; a
    fault that persists re-alarms on the next datum.
    """
    n = len(dev)
    out = {name: np.zeros(n) for name in ("ewma", "ewma_limit", "cusum_pos", "cusum_neg")}
    alarm = np.zeros(n, dtype=bool)
    e = sp = sm = 0.0
    since = 0  # datums since last restart
    for i, d in enumerate(dev):
        since += 1
        e = lam * d + (1 - lam) * e  # restarts from the target, as the limit formula assumes
        sp = max(0.0, sp + d / sigma - k)
        sm = max(0.0, sm - d / sigma - k)
        limit = 3.0 * sigma * np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * since)))
        out["ewma"][i], out["ewma_limit"][i], out["cusum_pos"][i], out["cusum_neg"][i] = e, limit, sp, sm
        alarm[i] = abs(d) > shewhart * sigma or abs(e) > limit or sp > h or sm > h
        if alarm[i]:
            e = sp = sm = 0.0
            since = 0
    return {**out, "alarm": alarm}


def pelt(x: np.ndarray, sigma: float, penalty: float | None = None, min_size: int = 2) -> list[int]:
    """PELT (Killick et al. 2012) for changes in mean with known noise sigma.

    Returns the indices where new segments start (excluding 0).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    beta = penalty if penalty is not None else 2.0 * np.log(n) * sigma**2 * 2
    cs = np.r_[0.0, np.cumsum(x)]
    cs2 = np.r_[0.0, np.cumsum(x**2)]

    def cost(a: int, b: int) -> float:  # segment x[a:b]
        m = b - a
        return float(cs2[b] - cs2[a] - (cs[b] - cs[a]) ** 2 / m)

    f = np.full(n + 1, np.inf)
    f[0] = -beta
    last = np.zeros(n + 1, dtype=int)
    candidates = [0]
    for t in range(min_size, n + 1):
        vals = [f[s] + cost(s, t) + beta if t - s >= min_size else np.inf for s in candidates]
        best = int(np.argmin(vals))
        f[t], last[t] = vals[best], candidates[best]
        candidates = [s for s, v in zip(candidates, vals, strict=True) if v - beta <= f[t]] + [t - min_size + 1]
    cps, t = [], n
    while t > 0:
        t = last[t]
        if t > 0:
            cps.append(t)
    return sorted(cps)


@dataclass
class Correlation:
    """CFD/surrogate -> tunnel mapping, CL_wt = k * CL_model + b (Huber fit)."""

    k: float
    b: float
    sigma: float

    @classmethod
    def fit(cls, cl_model: np.ndarray, cl_wt: np.ndarray) -> Correlation:
        reg = HuberRegressor().fit(cl_model[:, None], cl_wt)
        resid = cl_wt - reg.predict(cl_model[:, None])
        return cls(float(reg.coef_[0]), float(reg.intercept_), robust_sigma(resid))

    def predict(self, cl_model: np.ndarray) -> np.ndarray:
        return self.k * cl_model + self.b


@dataclass
class QAResult:
    flags: pd.DataFrame  # per-sample boolean flags by detector
    run_table: pd.DataFrame  # per-run statistics
    datum_table: pd.DataFrame  # per-datum-run control-chart statistics
    correlation: Correlation
    change_points: list[int]  # run indices where a level change starts
    tap_scores: np.ndarray  # [n_samples, n_taps] standardised reconstruction error


def run_qa(
    samples: pd.DataFrame,
    taps: np.ndarray,
    *,
    baseline_fraction: float = 0.2,
    ewma_lambda: float = 0.3,
    cusum_k: float = 0.5,
    cusum_h: float = 4.0,
) -> QAResult:
    df = samples.copy().reset_index(drop=True)
    n = len(df)
    base = df.index < int(baseline_fraction * n)

    corr = Correlation.fit(df.loc[base, "cl_model"].to_numpy(), df.loc[base, "cl"].to_numpy())
    df["resid"] = df["cl"] - corr.predict(df["cl_model"].to_numpy())

    # Spikes: Hampel with the baseline noise level as sigma (robust to the
    # rolling window landing inside a level shift).
    spike = hampel(df["resid"].to_numpy(), window=4, n_sigma=5.0, sigma=corr.sigma)

    # Run means with spikes removed.
    clean = df.loc[~spike]
    runs = clean.groupby("run").agg(
        run_type=("run_type", "first"), resid=("resid", "mean"), n=("resid", "size"), first=("sample", "min"), last=("sample", "max")
    )
    runs = runs.join(df.groupby("run")["sample"].agg(["min", "max"]).rename(columns={"min": "start", "max": "end"}))
    sigma_run = corr.sigma / np.sqrt(runs["n"].median())

    # Repeatability: control charts on datum runs only.
    datum = runs[runs["run_type"] == "datum"].copy()
    datum_ref = datum.loc[datum["end"] < int(baseline_fraction * n), "resid"]
    mu0 = float(datum_ref.mean()) if len(datum_ref) else 0.0
    sd0 = max(float(datum_ref.std(ddof=1)) if len(datum_ref) > 2 else sigma_run, sigma_run)
    dev = datum["resid"].to_numpy() - mu0
    chart = datum_chart(dev, sd0, ewma_lambda, cusum_k, cusum_h)
    for col, values in chart.items():
        datum[col] = values

    # Change points on all run means (datum + test), in run order.
    cps_idx = pelt(runs["resid"].to_numpy(), sigma=float(max(sigma_run, robust_sigma(np.diff(runs["resid"].to_numpy())) / np.sqrt(2))))
    change_runs = [int(runs.index[i]) for i in cps_idx]

    # Configuration outliers: a run whose mean residual is far from both
    # neighbours' level (isolated), as opposed to a sustained shift.
    level = runs["resid"].rolling(5, center=True, min_periods=2).median()
    runs["isolated_z"] = (runs["resid"] - level) / sigma_run
    # A sustained shift moves the neighbours too; an isolated run does not. So
    # the neighbour test is relative to the run's own deviation, not a fixed band.
    dev = (runs["resid"] - level).abs()
    tol = np.maximum(4 * sigma_run, 0.3 * dev)
    neighbours_ok = ((runs["resid"].shift(1) - level).abs() < tol) & ((runs["resid"].shift(-1) - level).abs() < tol)
    runs["config_outlier"] = (runs["isolated_z"].abs() > 6) & neighbours_ok & (runs["run_type"] == "test")

    # Pressure taps: PCA model of healthy tap vectors; a blocked tap has a
    # persistent large reconstruction error on its own channel.
    tb = taps[base]
    mu_t, sd_t = tb.mean(0), tb.std(0) + 1e-9
    zb = (tb - mu_t) / sd_t
    _, s, vt = np.linalg.svd(zb, full_matrices=False)
    k = int(np.searchsorted(np.cumsum(s**2) / np.sum(s**2), 0.99) + 1)
    p = vt[:k]
    z = (taps - mu_t) / sd_t
    recon_err = z - (z @ p.T) @ p
    err_sd = np.std(recon_err[base], axis=0) + 1e-9
    tap_scores = recon_err / err_sd
    tap_flag_raw = pd.DataFrame(np.abs(tap_scores) > 4).rolling(10, min_periods=5).mean().to_numpy() > 0.6
    tap_flag = tap_flag_raw.any(axis=1)

    # Per-sample flags.
    flags = pd.DataFrame(index=df.index)
    flags["spike"] = spike
    flags["drift_or_step"] = False
    alarm_datums = datum.index[datum["alarm"]]
    for r in alarm_datums:
        # The chart says "something changed since the previous datum"; a
        # change point inside that window says when, otherwise flag it all.
        prev = datum.index[datum.index < r]
        lo = int(prev[-1]) if len(prev) else int(runs.index[0]) - 1
        onset = [cp for cp in change_runs if lo < cp <= r]
        start = runs.loc[onset[0], "start"] if onset else (runs.loc[lo, "end"] + 1 if len(prev) else runs.loc[r, "start"])
        flags.loc[start : runs.loc[r, "end"], "drift_or_step"] = True
    flags["config_outlier"] = False
    for r in runs.index[runs["config_outlier"]]:
        flags.loc[runs.loc[r, "start"] : runs.loc[r, "end"], "config_outlier"] = True
    flags["tap"] = tap_flag
    return QAResult(flags=flags, run_table=runs, datum_table=datum, correlation=corr, change_points=change_runs, tap_scores=tap_scores)


def score(flags: np.ndarray, truth: np.ndarray, intervals: list[tuple[int, int]]) -> dict[str, float]:
    """Event recall, mean detection delay (samples) and sample-level precision."""
    detected, delays = 0, []
    for a, b in intervals:
        hit = np.flatnonzero(flags[a : b + 1])
        if hit.size:
            detected += 1
            delays.append(int(hit[0]))
    tp = int(np.sum(flags & truth))
    fp = int(np.sum(flags & ~truth))
    return {
        "events": len(intervals),
        "recall": detected / len(intervals) if intervals else float("nan"),
        "mean_delay": float(np.mean(delays)) if delays else float("nan"),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "false_alarms": fp,
    }
