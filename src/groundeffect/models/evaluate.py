"""Model evaluation and release gates.

A model is only promoted if it beats explicit, versioned thresholds on an
independent test design (Latin hypercube, never seen in training), including
uncertainty calibration. Thresholds live in ``configs/gates.yaml`` so a change
to the bar is a reviewed code change, not a silent edit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import kendalltau, norm, spearmanr

from groundeffect.models.data import SCALARS


def scalar_metrics(y: np.ndarray, pred: np.ndarray, std: np.ndarray | None = None) -> dict[str, float]:
    err = pred - y
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    out = {
        "r2": 1.0 - ss_res / ss_tot,
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "max_abs": float(np.max(np.abs(err))),
        "mape": float(np.mean(np.abs(err) / np.maximum(np.abs(y), 1e-9))),
    }
    if std is not None:
        z = np.abs(err) / np.maximum(std, 1e-12)
        out["coverage_1s"] = float(np.mean(z <= 1.0))
        out["coverage_2s"] = float(np.mean(z <= 2.0))
        out["nll"] = float(np.mean(-norm.logpdf(err, scale=np.maximum(std, 1e-12))))
    return out


def delta_metrics(y: np.ndarray, pred: np.ndarray, min_delta: float = 0.0) -> dict[str, float]:
    """How well the model ranks variants, which is what a design decision uses.

    Over all pairs of test designs whose true difference exceeds ``min_delta``
    (e.g. twice the CFD noise floor), the fraction where the predicted delta has
    the right sign, the RMSE of the predicted deltas, and rank correlations.
    """
    i, j = np.triu_indices(len(y), k=1)
    d_true, d_pred = y[i] - y[j], pred[i] - pred[j]
    keep = np.abs(d_true) > min_delta
    return {
        "pair_sign_accuracy": float(np.mean(np.sign(d_pred[keep]) == np.sign(d_true[keep]))),
        "delta_rmse": float(np.sqrt(np.mean((d_pred - d_true) ** 2))),
        "spearman": float(spearmanr(y, pred).statistic),
        "kendall": float(kendalltau(y, pred).statistic),
        "n_pairs": float(keep.sum()),
    }


def evaluate(
    y: np.ndarray, pred: np.ndarray, std: np.ndarray, cp: np.ndarray, cp_pred: np.ndarray, min_delta: float = 0.0
) -> dict[str, dict[str, float]]:
    report = {name: scalar_metrics(y[:, j], pred[:, j], std[:, j]) for j, name in enumerate(SCALARS)}
    for j, name in enumerate(SCALARS[:2]):
        report[name].update(delta_metrics(y[:, j], pred[:, j], min_delta if name == "cl_down" else 0.0))
    report["cp"] = {
        "rmse": float(np.sqrt(np.mean((cp_pred - cp) ** 2))),
        "max_abs": float(np.max(np.abs(cp_pred - cp))),
        "rmse_worst_case": float(np.max(np.sqrt(np.mean((cp_pred - cp) ** 2, axis=1)))),
    }
    return report


@dataclass(frozen=True)
class GateResult:
    name: str
    value: float
    threshold: float
    op: str
    passed: bool


def check_gates(report: dict[str, dict[str, float]], gates_file: Path) -> list[GateResult]:
    """Gates file format: ``{quantity: {metric: {min|max: value}}}``."""
    spec = yaml.safe_load(gates_file.read_text())
    results = []
    for qty, metrics in spec.items():
        for metric, bound in metrics.items():
            value = report[qty][metric]
            for op, thr in bound.items():
                ok = value >= thr if op == "min" else value <= thr
                results.append(GateResult(f"{qty}.{metric}", value, float(thr), op, bool(ok)))
    return results


def numerical_noise_floor(grid_study: Path, quantity: str = "cl_down") -> float | None:
    """Mean absolute discretisation uncertainty of the medium (production) mesh."""
    import json

    if not grid_study.exists():
        return None
    rows = [r for r in json.loads(grid_study.read_text()) if r["quantity"] == quantity]
    if not rows:
        return None
    return float(np.mean([r["gci_medium"] * abs(r["phi"][1]) for r in rows]))
