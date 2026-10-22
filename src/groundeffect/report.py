"""Figures for the README and the model report (static PNG, light surface)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Validated reference palette (dataviz skill): fixed slot order, never cycled.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GOOD, CRITICAL, WARNING = "#0ca30c", "#d03b3b", "#fab219"
CFD, SURROGATE = BLUE, ORANGE

DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#104281", "#2a78d6", "#9ec5f4", "#f0efec", "#f3a5a4", "#e34948", "#8f1f1f"]
)
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_seq", ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "legend.frameon": False,
            "legend.labelcolor": INK_2,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
        }
    )


def _log_ride_height_axis(ax: plt.Axes) -> None:
    from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(ScalarFormatter())


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def design_space(x: np.ndarray, status: np.ndarray, split: np.ndarray, path: Path) -> Path:
    style()
    names = ["ride height h/c", "incidence α [deg]", "camber m", "thickness t"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    pairs = [(0, 1), (0, 2), (1, 3)]
    for ax, (i, j) in zip(axes, pairs, strict=True):
        for name, color, marker in (("train", BLUE, "o"), ("test", ORANGE, "D")):
            m = (split == name) & (status == "ok")
            ax.scatter(x[m, i], x[m, j], s=14, color=color, marker=marker, alpha=0.8, linewidths=0, label=f"{name} (ok)")
        bad = status != "ok"
        ax.scatter(x[bad, i], x[bad, j], s=40, facecolors="none", edgecolors=CRITICAL, linewidths=1.4, marker="X", label="QA rejected")
        ax.set_xlabel(names[i])
        ax.set_ylabel(names[j])
        if i == 0:
            _log_ride_height_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, fontsize=8, bbox_to_anchor=(0.99, 1.04))
    title = f"Design of experiments — {int((status == 'ok').sum())} usable / {len(status)} CFD cases"
    fig.suptitle(title, x=0.01, y=1.02, ha="left", fontsize=11, fontweight="bold", color=INK)
    return _save(fig, path)


def parity(y: np.ndarray, pred: np.ndarray, std: np.ndarray, labels: list[str], metrics: list[dict[str, float]], path: Path) -> Path:
    style()
    fig, axes = plt.subplots(1, len(labels), figsize=(4.2 * len(labels), 4))
    for k, ax in enumerate(np.atleast_1d(axes)):
        lo, hi = min(y[:, k].min(), pred[:, k].min()), max(y[:, k].max(), pred[:, k].max())
        pad = 0.05 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=AXIS, lw=1, zorder=0)
        ax.errorbar(
            y[:, k],
            pred[:, k],
            yerr=2 * std[:, k],
            fmt="o",
            ms=5,
            color=SURROGATE,
            ecolor=SURROGATE,
            elinewidth=0.8,
            alpha=0.85,
            mec=SURFACE,
            mew=0.8,
        )
        m = metrics[k]
        ax.set_title(labels[k])
        ax.text(
            0.04,
            0.96,
            f"R² = {m['r2']:.4f}\nMAPE = {100 * m['mape']:.2f}%\n2σ coverage = {100 * m['coverage_2s']:.0f}%",
            transform=ax.transAxes,
            va="top",
            color=INK_2,
            fontsize=9,
        )
        ax.set_xlabel("CFD (held-out test design)")
        ax.set_ylabel("surrogate ± 2σ")
    return _save(fig, path)


def cp_examples(x_c: np.ndarray, cases: list[dict[str, object]], path: Path) -> Path:
    """cases: dicts with title, cp (160), pred (160), std (160)."""
    style()
    fig, axes = plt.subplots(1, len(cases), figsize=(4.4 * len(cases), 3.8), sharey=True)
    n = len(x_c)
    for ax, c in zip(np.atleast_1d(axes), cases, strict=True):
        cp, pred, std = (np.asarray(c[k]) for k in ("cp", "pred", "std"))
        for sl, side in ((slice(0, n), "suction"), (slice(n, 2 * n), "pressure")):
            ax.fill_between(x_c, pred[sl] - 2 * std[sl], pred[sl] + 2 * std[sl], color=SURROGATE, alpha=0.18, lw=0)
            ax.plot(x_c, pred[sl], color=SURROGATE, lw=1.6, label="surrogate ± 2σ" if side == "suction" else None)
            ax.plot(x_c, cp[sl], "o", color=CFD, ms=3, label="CFD" if side == "suction" else None)
        ax.annotate(
            "suction side",
            (0.7, float(np.interp(0.7, x_c, cp[:n]))),
            xytext=(0, 12),
            textcoords="offset points",
            color=INK_2,
            fontsize=8,
            ha="center",
        )
        ax.annotate(
            "pressure side",
            (0.7, float(np.interp(0.7, x_c, cp[n:]))),
            xytext=(0, -18),
            textcoords="offset points",
            color=INK_2,
            fontsize=8,
            ha="center",
        )
        ax.set_title(str(c["title"]), fontsize=10)
        ax.set_xlabel("x/c")
    np.atleast_1d(axes)[0].set_ylabel("Cp")
    np.atleast_1d(axes)[0].invert_yaxis()
    np.atleast_1d(axes)[0].legend(loc="lower right", fontsize=8)
    return _save(fig, path)


def ride_height_sweep(
    h_cfd: np.ndarray,
    cl_cfd: np.ndarray,
    ok: np.ndarray,
    h: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
    gp_mu: np.ndarray,
    gp_sd: np.ndarray,
    path: Path,
    title: str,
) -> Path:
    style()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.fill_between(h, mu - 2 * sd, mu + 2 * sd, color=SURROGATE, alpha=0.18, lw=0)
    ax.plot(h, mu, color=SURROGATE, label="PyTorch ensemble ± 2σ")
    ax.plot(h, gp_mu, color=AQUA, lw=1.6, ls="--", label="Gaussian process")
    ax.fill_between(h, gp_mu - 2 * gp_sd, gp_mu + 2 * gp_sd, color=AQUA, alpha=0.10, lw=0)
    ax.plot(h_cfd[ok], cl_cfd[ok], "o", color=CFD, ms=7, mec=SURFACE, mew=1.2, label="CFD, unseen sweep", zorder=5)
    if (~ok).any():
        ax.plot(h_cfd[~ok], cl_cfd[~ok], "X", color=CRITICAL, ms=8, label="CFD, QA rejected (unsteady)", zorder=5)
    _log_ride_height_axis(ax)
    ax.set_xlabel("ride height h/c")
    ax.set_ylabel("downforce coefficient CL")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    return _save(fig, path)


def learning_curves(curves: dict[str, list[tuple[int, float]]], noise_floor: float | None, path: Path) -> Path:
    style()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = {"max_std": ORANGE, "sequence": BLUE, "random": AQUA}
    labels = {"max_std": "active learning (max σ)", "sequence": "Sobol order", "random": "random"}
    for name, curve in curves.items():
        n, e = np.array(curve).T
        ax.plot(n, e, color=colors[name], marker="o", ms=4, label=labels[name])
    if noise_floor:
        ax.axhline(noise_floor, color=MUTED, lw=1, ls=":")
        ax.text(ax.get_xlim()[1], noise_floor, " CFD discretisation\n uncertainty (GCI)", va="center", ha="left", color=INK_2, fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("number of CFD runs used for training")
    ax.set_ylabel("test RMSE of CL (downforce)")
    ax.set_title("How many CFD runs does a GP surrogate need?")
    ax.legend(fontsize=8)
    return _save(fig, path)


def grid_convergence(rows: list[dict[str, Any]], path: Path) -> Path:
    style()
    RIDE = {"low": 0.10, "mid": 0.20, "high": 0.50}  # noqa: N806
    fig, ax = plt.subplots(figsize=(6.5, 4))
    colors = [BLUE, ORANGE, AQUA]
    for color, r in zip(colors, [r for r in rows if r["quantity"] == "cl_down"], strict=False):
        h = np.array(r["h"], dtype=float)
        phi = np.array(r["phi"], dtype=float)
        ax.plot(
            h / h[1],
            phi,
            "o-",
            color=color,
            label=f"h/c = {RIDE[str(r['design'])]}: GCI(medium) = {100 * float(r['gci_medium']):.1f}%, {'oscillatory' if r['oscillatory'] else 'monotonic'}",
        )  # type: ignore[arg-type]
        ax.plot([0], [r["phi_extrapolated"]], "*", color=color, ms=10)
    ax.set_xlim(left=-0.05)
    ax.set_xlabel("relative cell size h / h_medium  (★ = Richardson extrapolation, h → 0)")
    ax.set_ylabel("downforce coefficient CL")
    ax.set_title("Grid convergence (Celik et al. 2008)")
    ax.legend(fontsize=8)
    return _save(fig, path)


def wind_tunnel_qa(samples, qa, faults, path: Path) -> Path:  # type: ignore[no-untyped-def]
    style()
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.5), gridspec_kw={"height_ratios": [2.2, 1.4, 1.2]}, sharex=False)
    ax = axes[0]
    resid = samples["cl"].to_numpy() - qa.correlation.predict(samples["cl_model"].to_numpy())
    t = samples["sample"].to_numpy()
    for f in faults:
        if f.kind != "spike":
            ax.axvspan(f.start, f.end, color=GRID, alpha=0.9, lw=0)
            ax.text((f.start + f.end) / 2, 0.052, f.kind.replace("_", " "), ha="center", color=INK_2, fontsize=8)
    ax.plot(t, resid, color=BLUE, lw=0.8, label="CL residual vs surrogate (after CFD→tunnel correlation)")
    flags = qa.flags
    for col, color, marker, label in (("spike", CRITICAL, "v", "spike"), ("config_outlier", WARNING, "s", "configuration outlier")):
        m = flags[col].to_numpy()
        # Clip to the axis so off-scale spikes and outliers stay visible.
        ax.plot(t[m], np.clip(resid[m], -0.056, 0.056), marker, color=color, ms=6, ls="none", label=f"flagged: {label}", mec=INK, mew=0.4)
    m = flags["drift_or_step"].to_numpy()
    ax.plot(t[m], np.full(m.sum(), -0.045), "|", color=ORANGE, ms=8, label="flagged: drift / step")
    ax.set_ylim(-0.06, 0.06)
    ax.set_ylabel("ΔCL")
    ax.set_title("Wind-tunnel campaign QA — injected faults (grey) vs detections")
    ax.legend(fontsize=8, ncol=4, loc="upper left", bbox_to_anchor=(0.0, -0.07))

    ax = axes[1]
    d = qa.datum_table
    ax.plot(d.index, d["ewma"], color=BLUE, marker="o", ms=4, label="EWMA of datum-run deviation")
    ax.plot(d.index, d["ewma_limit"], color=MUTED, lw=1, ls="--", label="EWMA control limit")
    ax.plot(d.index, -d["ewma_limit"], color=MUTED, lw=1, ls="--")
    al = d["alarm"].to_numpy()
    ax.plot(d.index[al], d["ewma"][al], "o", color=CRITICAL, ms=7, mfc="none", mew=1.5, label="alarm (Shewhart / EWMA / CUSUM)")
    for cp in qa.change_points:
        ax.axvline(cp, color=AQUA, lw=0.8, alpha=0.7)
    ax.set_xlabel("run number (vertical lines: PELT change points)")
    ax.set_ylabel("ΔCL (datum)")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[2]
    im = ax.imshow(np.abs(qa.tap_scores).T, aspect="auto", cmap=SEQUENTIAL, vmin=0, vmax=8, interpolation="nearest")
    ax.set_ylabel("pressure tap")
    ax.set_xlabel("sample")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("|PCA residual| / σ", color=INK_2)
    fig.tight_layout()
    return _save(fig, path)


def field_comparison(
    grid_true: np.ndarray, grid_pred: np.ndarray, extent: tuple[float, float, float, float], title: str, path: Path
) -> Path:
    """Cp field: CFD vs model vs absolute error, on the gridded window."""
    style()
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 7.6), layout="constrained")
    vmax = float(np.nanpercentile(np.abs(grid_true), 99))
    for ax, data, name, cmap, vmin in (
        (axes[0], grid_true, "CFD (OpenFOAM)", DIVERGING, -vmax),
        (axes[1], grid_pred, "MeshGraphNet surrogate", DIVERGING, -vmax),
        (axes[2], np.abs(grid_pred - grid_true), "|error|", SEQUENTIAL, 0.0),
    ):
        im = ax.imshow(
            data, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax if name != "|error|" else vmax / 4, aspect="equal"
        )
        ax.set_title(f"{name} — Cp", fontsize=10)
        ax.grid(False)
        ax.set_ylabel("y/c")
        fig.colorbar(im, ax=ax, pad=0.01, shrink=0.9)
    axes[-1].set_xlabel("x/c")
    fig.suptitle(title, fontsize=10.5, fontweight="bold", color=INK)
    return _save(fig, path)


def flow_field(grid: np.ndarray, extent: tuple[float, float, float, float], title: str, path: Path) -> Path:
    style()
    fig, ax = plt.subplots(figsize=(10, 4))
    cp = grid[..., 2]
    vmax = float(np.nanpercentile(np.abs(cp), 99))
    im = ax.imshow(cp, origin="lower", extent=extent, cmap=DIVERGING, vmin=-vmax, vmax=vmax, aspect="equal")
    ny, nx = cp.shape
    xs, ys = np.linspace(extent[0], extent[1], nx), np.linspace(extent[2], extent[3], ny)
    u, v = np.nan_to_num(grid[..., 0]), np.nan_to_num(grid[..., 1])
    seeds = np.column_stack([np.full(34, extent[0] + 0.01), np.linspace(extent[2] + 0.01, extent[3] - 0.02, 34)])
    ax.streamplot(xs, ys, u, v, color=INK_2, linewidth=0.5, start_points=seeds, arrowstyle="-", broken_streamlines=False)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.grid(False)
    ax.set_title(title)
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c (ground at 0)")
    fig.colorbar(im, ax=ax, pad=0.01, label="Cp")
    return _save(fig, path)


def mesh_zoom(
    nodes: np.ndarray, tris: np.ndarray, quads: np.ndarray, box: tuple[float, float, float, float], title: str, path: Path
) -> Path:
    style()
    from matplotlib.collections import PolyCollection

    fig, ax = plt.subplots(figsize=(10, 4.2))
    polys = [nodes[t] for t in tris] + [nodes[q] for q in quads]
    colors = [SURFACE] * len(tris) + ["#cde2fb"] * len(quads)
    ax.add_collection(PolyCollection(polys, facecolors=colors, edgecolors=INK_2, linewidths=0.15))
    ax.set_xlim(box[0], box[1])
    ax.set_ylim(box[2], box[3])
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m] (moving ground at 0)")
    return _save(fig, path)
