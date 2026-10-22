"""Command line entry point: ``groundeffect --help``."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from groundeffect.doe import PARAMS, DesignSpace, latin_hypercube, sobol
from groundeffect.foam.case import FlowSpec
from groundeffect.foam.runner import FoamRunner
from groundeffect.geometry import WingDesign
from groundeffect.meshing import MeshSpec
from groundeffect.pipeline import CaseResult, run_batch, run_case

app = typer.Typer(add_completion=False, no_args_is_help=True, help="CFD -> ML pipeline for a wing in ground effect.")

RUNS = Path("runs")
DATA = Path("data")


def _log(r: CaseResult) -> None:
    typer.echo(
        f"[{r.status:>13}] {r.case_id}  h/c={r.design['h_c']:.3f} a={r.design['alpha_deg']:.2f} "
        f"m={r.design['camber']:.3f} t={r.design['thickness']:.3f}  CLd={r.cl_down:.4f} CD={r.cd:.4f} "
        f"cells={r.n_cells} {r.wall_time_s:.0f}s"
    )


@app.command()
def smoke(
    out: Path = RUNS / "smoke",
    refinement: float = 0.6,
    iterations: int = 600,
    image: str | None = None,
    mode: str = "auto",
) -> None:
    """Run one coarse case end to end and check it against physical bounds (used in CI)."""
    runner = FoamRunner(mode=mode, image=image) if image else FoamRunner(mode=mode)
    r = run_case(
        WingDesign(h_c=0.15, alpha_deg=4.0, camber=0.04, thickness=0.12),
        out,
        mesh=MeshSpec(refinement=refinement),
        flow=FlowSpec(end_iter=iterations),
        runner=runner,
    )
    _log(r)
    checks = {
        "solver ran": r.status in ("ok", "unconverged"),
        "mesh ok": r.mesh_ok,
        "downforce 0.9 < CL_down < 1.6": 0.9 < r.cl_down < 1.6,
        "drag 0.01 < CD < 0.08": 0.01 < r.cd < 0.08,
        "wing y+ mean < 5": r.yplus_wing_mean < 5,
    }
    fields = np.load(out / "fields.npz")
    cn = np.trapezoid(fields["cp_pressure"] - fields["cp_suction"], fields["x_c"])
    checks["Cp integral matches forces (5%)"] = abs(cn - r.cl_down) / r.cl_down < 0.05
    for name, ok in checks.items():
        typer.echo(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not all(checks.values()):
        raise typer.Exit(1)


@app.command("run-doe")
def run_doe(
    n_train: int = 256,
    n_test: int = 48,
    workers: int = 12,
    root: Path = RUNS / "doe",
    seed: int = 0,
) -> None:
    """Run the Sobol training set and an independent Latin-hypercube test set."""
    space = DesignSpace()
    jobs = [(f"train_{i:03d}", d) for i, d in enumerate(space.designs(sobol(space, n_train, seed)))]
    jobs += [(f"test_{i:03d}", d) for i, d in enumerate(space.designs(latin_hypercube(space, n_test, seed + 1)))]
    results = run_batch(jobs, root, workers=workers, on_result=_log)
    statuses = pd.Series([r.status for r in results]).value_counts()
    typer.echo(statuses.to_string())


@app.command("grid-study")
def grid_study(
    root: Path = RUNS / "gci",
    refinements: str = "0.707,1.0,1.414",
    workers: int = 9,
) -> None:
    """Three-level grid convergence study at low, mid and high ride height."""
    designs = {
        "low": WingDesign(h_c=0.10, alpha_deg=4.0, camber=0.04, thickness=0.12),
        "mid": WingDesign(h_c=0.20, alpha_deg=4.0, camber=0.04, thickness=0.12),
        "high": WingDesign(h_c=0.50, alpha_deg=4.0, camber=0.04, thickness=0.12),
    }
    levels = [float(x) for x in refinements.split(",")]
    from concurrent.futures import ThreadPoolExecutor

    from groundeffect.verification import gci

    def one(name: str, r: float) -> CaseResult:
        case = root / f"{name}_r{r:.3f}"
        if (case / "result.json").exists():  # reuse finished runs
            return CaseResult(**json.loads((case / "result.json").read_text()))
        return run_case(designs[name], case, mesh=MeshSpec(refinement=r), flow=FlowSpec(end_iter=2500), keep_fields=False)

    with ThreadPoolExecutor(workers) as pool:
        res = {(n, r): f.result() for (n, r), f in {(n, r): pool.submit(one, n, r) for n in designs for r in levels}.items()}
    rows = []
    for name in designs:
        fine, med, coarse = (res[(name, r)] for r in sorted(levels, reverse=True))
        if any(c.status != "ok" for c in (fine, med, coarse)):
            typer.echo(f"{name:>5}: skipped, not every grid level converged (GCI needs converged solutions)")
            continue
        for qty in ("cl_down", "cd"):
            g = gci(
                (getattr(fine, qty), getattr(med, qty), getattr(coarse, qty)),
                (fine.n_cells, med.n_cells, coarse.n_cells),
            )
            rows.append({"design": name, "quantity": qty, **asdict(g)})
            typer.echo(
                f"{name:>5} {qty:>8}: fine={g.phi[0]:.4f} med={g.phi[1]:.4f} coarse={g.phi[2]:.4f} "
                f"p={g.order:.2f} ext={g.phi_extrapolated:.4f} GCI_med={100 * g.gci_medium:.2f}% "
                f"GCI_fine={100 * g.gci_fine:.2f}%"
            )
    DATA.mkdir(exist_ok=True)
    (DATA / "grid_study.json").write_text(json.dumps(rows, indent=2, default=float))


@app.command()
def retry(root: Path = RUNS / "doe", workers: int = 12) -> None:
    """Re-run every unconverged case in ``root`` with the robust solver profile."""
    from concurrent.futures import ThreadPoolExecutor

    from groundeffect.pipeline import retry_case

    todo = [rj.parent for rj in sorted(root.glob("*/result.json")) if json.loads(rj.read_text())["status"] == "unconverged"]
    typer.echo(f"{len(todo)} unconverged cases to retry")
    with ThreadPoolExecutor(workers) as pool:
        for r in pool.map(retry_case, todo):
            before = r.extra.get("cl_rel_std_before_retry", float("nan"))
            typer.echo(f"retry {r.case_id}: {r.status}  rel std {before:.1e} -> {r.cl_rel_std:.1e}  CLd={r.cl_down:.4f}")


@app.command("build-dataset")
def build_dataset(root: Path = RUNS / "doe", out: Path = DATA / "dataset") -> None:
    """Collect case results into a versioned, ML-ready dataset."""
    out.mkdir(parents=True, exist_ok=True)
    rows, cp_s, cp_p, ground, grids, ids = [], [], [], [], [], []
    for rj in sorted(root.glob("*/result.json")):
        r = json.loads(rj.read_text())
        rows.append(
            {
                "case_id": r["case_id"],
                "split": r["case_id"].split("_")[0],
                **r["design"],
                **{k: v for k, v in r.items() if k not in ("design", "extra", "case_id")},
            }
        )
        f = rj.parent / "fields.npz"
        if r["status"] == "ok" and f.exists():
            z = np.load(f)
            ids.append(r["case_id"])
            cp_s.append(z["cp_suction"])
            cp_p.append(z["cp_pressure"])
            ground.append(z["ground_cp"])
            grids.append(z["grid"].astype(np.float16))
            x_c, ground_x = z["x_c"], z["ground_x"]
    df = pd.DataFrame(rows)
    df.to_csv(out / "cases.csv", index=False)
    if not ids:
        raise typer.BadParameter("no usable cases found")
    np.savez_compressed(
        out / "surface.npz",
        case_id=np.array(ids),
        x_c=x_c,
        cp_suction=np.array(cp_s),
        cp_pressure=np.array(cp_p),
        ground_x=ground_x,
        ground_cp=np.array(ground),
    )
    np.savez_compressed(out / "grid.npz", case_id=np.array(ids), grid=np.array(grids))
    summary = df.groupby(["split", "status"]).size().to_dict()
    (out / "manifest.json").write_text(
        json.dumps({"params": list(PARAMS), "counts": {f"{k[0]}/{k[1]}": int(v) for k, v in summary.items()}}, indent=2)
    )
    typer.echo(df.groupby(["split", "status"]).size().to_string())


ARTIFACTS = Path("artifacts")
FIGURES = Path("docs/figures")


def _dataset_hash(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    for f in sorted(path.glob("*")):
        if f.is_file():
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:12]


@app.command()
def train(
    dataset: Path = DATA / "dataset",
    out: Path = ARTIFACTS,
    gates: Path = Path("configs/gates.yaml"),
    epochs: int = 3000,
    members: int = 5,
    figures: bool = True,
) -> None:
    """Train GP + PyTorch ensemble, evaluate on the held-out design, apply release gates, export."""
    from groundeffect.models.data import SCALARS, load_dataset
    from groundeffect.models.ensemble import EnsembleSurrogate, TrainConfig, export_program
    from groundeffect.models.evaluate import check_gates, evaluate, numerical_noise_floor, scalar_metrics
    from groundeffect.models.gp import GPSurrogate

    ds = load_dataset(dataset)
    typer.echo(f"dataset {_dataset_hash(dataset)}: {len(ds.train.x)} train / {len(ds.test.x)} test (QA-passed cases)")
    sur = EnsembleSurrogate(config=TrainConfig(epochs=epochs, members=members)).fit(ds.train.x, ds.train.y, ds.train.cp)
    assert sur.pod is not None
    typer.echo(f"POD: {sur.pod.k} modes capture {100 * sur.pod.energy[-1]:.3f}% of Cp variance")
    pred = sur.predict(ds.test.x)
    floor = numerical_noise_floor(DATA / "grid_study.json")
    # Only pairs whose true CL difference exceeds 2x the CFD noise floor are a
    # meaningful ranking test; smaller deltas are inside the CFD's own error.
    report = evaluate(ds.test.y, pred["scalars"], pred["scalars_std"], ds.test.cp, pred["cp"], min_delta=2 * (floor or 0.0))

    gp = GPSurrogate().fit(ds.train.x, ds.train.y)
    gmu, gsd = gp.predict(ds.test.x)
    report["gp"] = {
        f"{name}_{k}": v for j, name in enumerate(SCALARS) for k, v in scalar_metrics(ds.test.y[:, j], gmu[:, j], gsd[:, j]).items()
    }
    report["reference"] = {
        "cl_down_gci_abs": floor or float("nan"),
        "n_train": len(ds.train.x),
        "n_test": len(ds.test.x),
        "pod_modes": sur.pod.k,
    }

    results = check_gates(report, gates)
    for g in results:
        typer.echo(f"  {'PASS' if g.passed else 'FAIL'}  {g.name:<24} {g.value:.4f} ({g.op} {g.threshold})")
    out.mkdir(parents=True, exist_ok=True)
    sur.save(out / "ensemble")
    export_program(sur, out / "ensemble.pt2")
    meta = {
        "dataset": _dataset_hash(dataset),
        "metrics": report,
        "gates": [g.__dict__ for g in results],
        "passed": all(g.passed for g in results),
        "design_space": {k: list(v) for k, v in zip(PARAMS, DesignSpace().bounds().tolist(), strict=True)},
    }
    (out / "metrics.json").write_text(json.dumps(meta, indent=2, default=float))
    _model_card(meta, out / "model_card.md")
    if figures:
        from groundeffect import report as fig

        labels = ["downforce CL", "drag CD", "pitching moment Cm"]
        fig.parity(ds.test.y, pred["scalars"], pred["scalars_std"], labels, [report[k] for k in SCALARS], FIGURES / "parity.png")
        err = np.sqrt(np.mean((pred["cp"] - ds.test.cp) ** 2, axis=1))
        picks = [int(np.argmin(np.abs(err - np.quantile(err, q)))) for q in (0.1, 0.5, 0.95)]
        cases = []
        for i, q in zip(picks, ("best decile", "median", "95th percentile"), strict=True):
            h, a, m, t = ds.test.x[i]
            cases.append(
                {
                    "title": f"{q}: h/c={h:.2f}, α={a:.1f}°, NACA {round(m * 100)}4{round(t * 100):02d}",
                    "cp": ds.test.cp[i],
                    "pred": pred["cp"][i],
                    "std": pred["cp_std"][i],
                }
            )
        fig.cp_examples(ds.x_c, cases, FIGURES / "cp_examples.png")
        cases_df = ds.cases
        fig.design_space(
            cases_df[list(PARAMS)].to_numpy(float),
            cases_df["status"].to_numpy(),
            cases_df["split"].to_numpy(),
            FIGURES / "design_space.png",
        )
    if not meta["passed"]:
        typer.echo("release gates FAILED: model not promoted")
        raise typer.Exit(1)
    typer.echo(f"model promoted -> {out / 'ensemble.pt2'}")


def _model_card(meta: dict, path: Path) -> None:  # type: ignore[type-arg]
    m = meta["metrics"]
    gate_rows = "\n".join(
        f"| `{g['name']}` | {g['value']:.4f} | {g['op']} {g['threshold']} | {'pass' if g['passed'] else 'FAIL'} |" for g in meta["gates"]
    )
    bounds = "\n".join(f"| `{k}` | {v[0]} | {v[1]} |" for k, v in meta["design_space"].items())
    path.write_text(
        f"""# Model card: ground-effect wing surrogate

**Dataset** `{meta["dataset"]}`: {m["reference"]["n_train"]} training / {m["reference"]["n_test"]} test CFD cases
(2D steady RANS, k-omega SST, OpenFOAM v2512, Re = 5e5, moving ground). Only cases that passed
convergence and mesh QA are used.

**Model**: deep ensemble of 5 MLPs (PyTorch) predicting CL (downforce), CD, Cm and {m["reference"]["pod_modes"]}
POD coefficients of surface Cp, with variance calibrated on a held-out split. Exported with `torch.export`.

## Release gates (independent Latin-hypercube test design)

| gate | value | threshold | result |
|---|---|---|---|
{gate_rows}

Downforce: R2 {m["cl_down"]["r2"]:.4f}, RMSE {m["cl_down"]["rmse"]:.4f}, max error {m["cl_down"]["max_abs"]:.4f}.
CFD discretisation uncertainty of CL on the production mesh (mean GCI): {m["reference"]["cl_down_gci_abs"]:.4f}.
GP baseline on the same split: CL R2 {m["gp"]["cl_down_r2"]:.4f}, RMSE {m["gp"]["cl_down_rmse"]:.4f}.

## Valid domain

| variable | min | max |
|---|---|---|
{bounds}

## Limitations

* 2D sections: no tip vortices, endplates, or 3D relief; trends, not car numbers.
* Steady RANS: cases with massive separation (lowest ride heights at high incidence and camber)
  are rejected by QA and are therefore *not represented*; the model must not be trusted there.
* Outside the bounds above the service flags the request as extrapolation.
"""
    )


@app.command("active-learning")
def active_learning(dataset: Path = DATA / "dataset", n_max: int = 96, seeds: int = 3) -> None:
    """Retrospective study: GP accuracy vs number of CFD runs for three sampling strategies."""
    from groundeffect import report as fig
    from groundeffect.models.data import load_dataset
    from groundeffect.models.evaluate import numerical_noise_floor
    from groundeffect.models.gp import active_learning_curve

    ds = load_dataset(dataset)
    curves: dict[str, list[tuple[int, float]]] = {}
    for strategy in ("sequence", "random", "max_std"):
        runs = [
            active_learning_curve(ds.train.x, ds.train.y, ds.test.x, ds.test.y, strategy=strategy, n_max=n_max, seed=s)
            for s in range(1 if strategy == "sequence" else seeds)
        ]
        n = [p[0] for p in runs[0]]
        curves[strategy] = [(n[i], float(np.mean([r[i][1] for r in runs]))) for i in range(len(n))]
        typer.echo(f"{strategy:>9}: " + "  ".join(f"n={a}:{b:.4f}" for a, b in curves[strategy][::4]))
    (DATA / "active_learning.json").write_text(json.dumps(curves, indent=1))
    fig.learning_curves(curves, numerical_noise_floor(DATA / "grid_study.json"), FIGURES / "active_learning.png")


@app.command()
def sweep(
    root: Path = RUNS / "sweep",
    alpha: float = 4.0,
    camber: float = 0.04,
    thickness: float = 0.12,
    n: int = 10,
    workers: int = 10,
) -> None:
    """CFD ride-height sweep at one configuration (never in training) vs both surrogates."""
    from groundeffect import report as fig
    from groundeffect.models.data import load_dataset
    from groundeffect.models.ensemble import EnsembleSurrogate
    from groundeffect.models.gp import GPSurrogate

    hs = np.geomspace(0.08, 0.6, n)
    jobs = [(f"h{h:.3f}", WingDesign(float(h), alpha, camber, thickness)) for h in hs]
    res = run_batch(jobs, root, workers=workers, keep_fields=True, on_result=_log)
    h_cfd = np.array([r.design["h_c"] for r in res])
    cl_cfd = np.array([r.cl_down for r in res])
    ok = np.array([r.status == "ok" for r in res])
    sur = EnsembleSurrogate.load(ARTIFACTS / "ensemble")
    ds = load_dataset(DATA / "dataset")
    gp = GPSurrogate().fit(ds.train.x, ds.train.y)
    hh = np.geomspace(0.08, 0.6, 120)
    xx = np.column_stack([hh, np.full_like(hh, alpha), np.full_like(hh, camber), np.full_like(hh, thickness)])
    p = sur.predict(xx)
    gmu, gsd = gp.predict(xx)
    title = f"Ground effect: CL vs ride height (α = {alpha:g}°, NACA {round(camber * 100)}4{round(thickness * 100):02d})"
    fig.ride_height_sweep(
        h_cfd, cl_cfd, ok, hh, p["scalars"][:, 0], p["scalars_std"][:, 0], gmu[:, 0], gsd[:, 0], FIGURES / "ride_height_sweep.png", title
    )
    pt = sur.predict(np.column_stack([h_cfd, np.full_like(h_cfd, alpha), np.full_like(h_cfd, camber), np.full_like(h_cfd, thickness)]))
    for hc, c, o, m_, s_ in zip(h_cfd, cl_cfd, ok, pt["scalars"][:, 0], pt["scalars_std"][:, 0], strict=True):
        typer.echo(f"h/c={hc:.3f}  CFD={c:.4f} ({'ok' if o else 'QA rejected'})  surrogate={m_:.4f} ± {2 * s_:.4f}  err={m_ - c:+.4f}")
    best = sur.predict(xx)["scalars"][:, 0].argmax()
    typer.echo(f"surrogate peak downforce at h/c = {hh[best]:.3f}")


@app.command("wt-demo")
def wt_demo(seed: int = 7, use_surrogate: bool = True) -> None:
    """Simulate a tunnel campaign (loads from the surrogate), inject faults, run QA, score it."""
    from groundeffect import report as fig
    from groundeffect.geometry import cp_stations
    from groundeffect.windtunnel.detect import run_qa, score
    from groundeffect.windtunnel.synth import analytic_wing, simulate_campaign

    load = analytic_wing
    if use_surrogate and (ARTIFACTS / "ensemble").exists():
        from groundeffect.models.ensemble import EnsembleSurrogate

        sur = EnsembleSurrogate.load(ARTIFACTS / "ensemble")
        x_c, taps_x = cp_stations(), np.linspace(0.05, 0.95, 12)

        def load(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # type: ignore[misc]
            p = sur.predict(x)
            cp = p["cp"]
            taps = np.hstack(
                [np.array([np.interp(taps_x, x_c, r[:80]) for r in cp]), np.array([np.interp(taps_x, x_c, r[80:]) for r in cp])]
            )
            return p["scalars"][:, 0], p["scalars"][:, 1], taps

    camp = simulate_campaign(load, seed=seed)
    qa = run_qa(camp.samples, camp.taps)
    mapping = {
        "spike": "spike",
        "drift": "drift_or_step",
        "step": "drift_or_step",
        "tap_blocked": "tap",
        "config_outlier": "config_outlier",
    }
    scores = {}
    for kind, col in mapping.items():
        truth = np.zeros(len(camp.samples), dtype=bool)
        for f in camp.faults:
            if mapping[f.kind] == col:
                truth[f.start : f.end + 1] = True
        scores[kind] = score(qa.flags[col].to_numpy(), truth, [(f.start, f.end) for f in camp.faults if f.kind == kind])
        s = scores[kind]
        typer.echo(f"{kind:>15}: recall {s['recall']:.2f}  precision {s['precision']:.2f}  false-alarm samples {s['false_alarms']}")
    c = qa.correlation
    typer.echo(f"learned CFD->tunnel correlation: CL_wt = {c.k:.4f} CL_model + {c.b:+.4f} (noise sigma {c.sigma:.4f})")
    DATA.mkdir(exist_ok=True)
    (DATA / "wind_tunnel_qa.json").write_text(
        json.dumps(
            {"seed": seed, "scores": scores, "correlation": c.__dict__, "change_point_runs": qa.change_points}, indent=2, default=float
        )
    )
    fig.wind_tunnel_qa(camp.samples, qa, camp.faults, FIGURES / "wind_tunnel_qa.png")


@app.command("train-mgn")
def train_mgn_cmd(
    root: Path = RUNS / "doe",
    n_train: int = 96,
    epochs: int = 25,
    hidden: int = 64,
    steps: int = 8,
    backend: str = "auto",
    out: Path = ARTIFACTS / "mgn",
) -> None:
    """Train the MeshGraphNet field surrogate (PhysicsNeMo if installed) on the OpenFOAM cell graphs."""
    from groundeffect.models.mgn import MGNConfig, load_graph, save_mgn, train_mgn

    graphs = [load_graph(d) for d in _usable(root, "train")[:n_train]]
    typer.echo(f"{len(graphs)} training graphs, mean {np.mean([len(g.pos) for g in graphs]):.0f} nodes")
    cfg = MGNConfig(backend=backend, hidden=hidden, steps=steps, epochs=epochs)
    model, norm = train_mgn(graphs, cfg)
    save_mgn(model, norm, cfg, out)
    mgn_report(root=root, model_dir=out)


def _usable(root: Path, split: str) -> list[Path]:
    dirs = []
    for rj in sorted(root.glob(f"{split}_*/result.json")):
        if json.loads(rj.read_text())["status"] == "ok" and (rj.parent / "fields.npz").exists():
            dirs.append(rj.parent)
    return dirs


@app.command("mgn-report")
def mgn_report(root: Path = RUNS / "doe", model_dir: Path = ARTIFACTS / "mgn") -> None:
    """Evaluate a saved MeshGraphNet on every usable test graph and draw the field figure."""
    from scipy.interpolate import LinearNDInterpolator

    from groundeffect import report as fig
    from groundeffect.models.mgn import load_graph, load_mgn, predict_graph, relative_l2
    from groundeffect.pipeline import GRAPH_X, GRID_SHAPE, GRID_X, GRID_Y

    model, norm, cfg = load_mgn(model_dir)
    test_dirs = _usable(root, "test")
    errs = np.array([relative_l2(predict_graph(model, norm, g), g.y) for g in map(load_graph, test_dirs)])
    summary = {
        k: {"median": float(np.median(errs[:, i])), "p90": float(np.quantile(errs[:, i], 0.9))} for i, k in enumerate(("u", "v", "cp"))
    }
    typer.echo(
        f"{len(test_dirs)} test graphs, relative L2: "
        + ", ".join(f"{k} median {v['median']:.3f} / p90 {v['p90']:.3f}" for k, v in summary.items())
    )
    meta = {"backend": cfg.backend, "epochs": cfg.epochs, "n_test": len(test_dirs), "relative_l2": summary, "loss_history": cfg.history}
    (model_dir / "metrics.json").write_text(json.dumps(meta, indent=2))

    # Median-error test case on the plotting grid, cropped to the graph window.
    i = int(np.argsort(errs[:, 2])[len(errs) // 2])
    g = load_graph(test_dirs[i])
    pred = predict_graph(model, norm, g).numpy()
    z = np.load(test_dirs[i] / "fields.npz")
    gx, gy = np.linspace(*GRID_X, GRID_SHAPE[1]), np.linspace(*GRID_Y, GRID_SHAPE[0])
    cols, rows = gx <= GRAPH_X[1], gy <= 0.9
    mx, my = np.meshgrid(gx[cols], gy[rows])
    grid_pred = LinearNDInterpolator(g.pos.numpy(), pred[:, 2])(np.column_stack([mx.ravel(), my.ravel()])).reshape(mx.shape)
    grid_true = z["grid"][..., 2].astype(float)[np.ix_(rows, cols)]
    grid_pred[np.isnan(grid_true)] = np.nan
    d = json.loads((test_dirs[i] / "result.json").read_text())["design"]
    title = f"PhysicsNeMo MeshGraphNet on an unseen design\nh/c = {d['h_c']:.2f}, α = {d['alpha_deg']:.1f}° (median-error test case): Cp relative L2 error {errs[i, 2]:.3f}"
    extent = (GRID_X[0], float(gx[cols][-1]), GRID_Y[0], float(gy[rows][-1]))
    fig.field_comparison(grid_true, grid_pred, extent, title, FIGURES / "mgn_field.png")


@app.command("figures")
def figures_cmd(case: Path = RUNS / "sweep" / "h0.157") -> None:
    """Static figures that do not need a trained model: grid convergence, mesh, flow field."""
    from groundeffect import report as fig
    from groundeffect.geometry import place_wing
    from groundeffect.pipeline import GRID_X, GRID_Y

    if (DATA / "grid_study.json").exists():
        fig.grid_convergence(json.loads((DATA / "grid_study.json").read_text()), FIGURES / "grid_convergence.png")
    d = WingDesign(h_c=0.15, alpha_deg=4.0, camber=0.04, thickness=0.12)
    nodes, tris, quads = _mesh_2d(d)
    w = place_wing(d)
    fig.mesh_zoom(
        nodes,
        tris,
        quads,
        (w.le[0] - 0.03, w.te[0] + 0.08, 0.0, 0.11),
        "Automated hybrid mesh: prism layers + triangles, gap-aware refinement (h/c = 0.15)",
        FIGURES / "mesh.png",
    )
    fz = case / "fields.npz"
    if fz.exists():
        grid = np.load(fz)["grid"].astype(float)
        d = json.loads((case / "result.json").read_text())
        title = f"OpenFOAM solution: Cp and streamlines (h/c = {d['design']['h_c']:.2f}, α = {d['design']['alpha_deg']:g}°, CL = {d['cl_down']:.3f})"
        fig.flow_field(grid, (*GRID_X, *GRID_Y), title, FIGURES / "flow_field.png")


def _mesh_2d(design: WingDesign) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mesh in-process and return the 2D face (z = 0) as nodes / triangles / quads."""
    import tempfile

    import gmsh

    from groundeffect.geometry import place_wing
    from groundeffect.meshing import build_mesh

    with tempfile.TemporaryDirectory() as tmp:
        msh = Path(tmp) / "m.msh"
        build_mesh(place_wing(design), msh)
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.open(str(msh))
            tags, coords, _ = gmsh.model.mesh.getNodes()
            xyz = coords.reshape(-1, 3)
            index = {int(t): i for i, t in enumerate(tags)}
            tris: list[np.ndarray] = []
            quads: list[np.ndarray] = []
            for dim, ent in gmsh.model.getEntities(2):
                bb = gmsh.model.getBoundingBox(dim, ent)
                if abs(bb[2]) > 1e-9 or abs(bb[5]) > 1e-9:
                    continue  # keep only the z = 0 face
                etypes, _, enodes = gmsh.model.mesh.getElements(dim, ent)
                for et, en in zip(etypes, enodes, strict=True):
                    k = {2: 3, 3: 4}.get(int(et))
                    if k:
                        conn = np.vectorize(index.get)(en.reshape(-1, k))
                        (tris if k == 3 else quads).extend(conn)
        finally:
            gmsh.finalize()
    return xyz[:, :2], np.array(tris), np.array(quads)


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Serve the exported model over HTTP (FastAPI)."""
    import uvicorn

    uvicorn.run("groundeffect.serve:app", host=host, port=port)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
