"""End-to-end CFD case execution: geometry -> mesh -> OpenFOAM -> QA'd results.

Every case produces a ``result.json`` with the design, integrated coefficients,
convergence and mesh-quality metrics and a QA status, plus a ``fields.npz``
with surface Cp, a gridded flow window and the cell graph used by the
graph-network surrogate. Downstream ML only ever sees cases whose status is
``ok``; everything else is kept and labelled, never silently dropped.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from matplotlib.path import Path as PolyPath
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree

from groundeffect.foam import parse
from groundeffect.foam.case import FlowSpec, extend_run, write_allrun, write_case, write_fv_solution
from groundeffect.foam.runner import FoamRunner
from groundeffect.geometry import PlacedWing, WingDesign, cp_stations, naca4, place_wing
from groundeffect.meshing import MeshSpec, build_mesh

_GMSH_LOCK = threading.Lock()  # gmsh keeps global state: one mesher at a time

# Gridded window in chord units, relative to the wing LE (x) and the ground (y).
GRID_X = (-0.5, 2.5)
GRID_Y = (0.0, 1.2)
GRID_SHAPE = (64, 160)  # (ny, nx)
GRAPH_X = (-0.6, 2.0)
GRAPH_Y = (0.0, 1.3)


@dataclass(frozen=True)
class QATolerances:
    window: int = 300  # iterations used to judge force stability
    max_rel_std: float = 2e-3
    max_rel_drift: float = 2e-3


@dataclass
class CaseResult:
    case_id: str
    design: dict[str, float]
    status: str  # ok | unconverged | mesh_failed | solver_failed
    cl_down: float = float("nan")  # downforce coefficient, -Cl
    cd: float = float("nan")
    cm: float = float("nan")
    efficiency: float = float("nan")  # cl_down / cd
    cl_rel_std: float = float("nan")
    cl_rel_drift: float = float("nan")
    iterations: int = 0
    n_cells: int = 0
    mesh_ok: bool = False
    max_non_ortho: float = float("nan")
    max_skewness: float = float("nan")
    yplus_wing_mean: float = float("nan")
    yplus_wing_max: float = float("nan")
    wall_time_s: float = 0.0
    refinement: float = 1.0
    message: str = ""
    extra: dict[str, float] = field(default_factory=dict)


def run_case(
    design: WingDesign,
    case_dir: Path,
    *,
    mesh: MeshSpec | None = None,
    flow: FlowSpec | None = None,
    runner: FoamRunner | None = None,
    qa: QATolerances | None = None,
    keep_fields: bool = True,
    retry: bool = True,
) -> CaseResult:
    mesh = mesh or MeshSpec()
    flow = flow or FlowSpec()
    runner = runner or FoamRunner()
    qa = qa or QATolerances()
    t0 = time.time()
    case_dir.mkdir(parents=True, exist_ok=True)
    wing = place_wing(design)
    res = CaseResult(case_id=case_dir.name, design=design.as_dict(), status="solver_failed", refinement=mesh.refinement)
    try:
        with _GMSH_LOCK:
            stats = build_mesh(wing, case_dir / "mesh.msh", mesh)
    except Exception as exc:  # geometry the mesher cannot handle is data too
        res.status, res.message = "mesh_failed", f"gmsh: {exc}"
        return _finish(res, case_dir, t0)
    write_case(case_dir, wing, flow, mesh.span)
    write_allrun(case_dir)
    proc = runner.run(case_dir)
    if (case_dir / "log.checkMesh").exists():
        cm = parse.parse_checkmesh(case_dir / "log.checkMesh")
        res.n_cells, res.mesh_ok = cm.cells or stats.n_cells, cm.ok
        res.max_non_ortho, res.max_skewness = cm.max_non_ortho, cm.max_skewness
    if proc.returncode != 0:
        res.message = (proc.stderr or proc.stdout)[-500:]
        return _finish(res, case_dir, t0)

    _assess(case_dir, res, qa, "log.simpleFoam")
    if res.status == "unconverged" and retry:
        res = _retry(case_dir, res, flow, runner, qa)
    if keep_fields and res.status in ("ok", "unconverged"):
        try:
            _extract_fields(case_dir, design, wing, flow)
        except Exception as exc:
            res.message = f"field extraction failed: {exc}"
    return _finish(res, case_dir, t0)


def _assess(case_dir: Path, res: CaseResult, qa: QATolerances, log: str) -> None:
    """Integrated loads over the last window plus the convergence verdict."""
    coeffs = parse.read_table(parse.latest_dat(case_dir, "forceCoeffs", "coefficient.dat"))
    cl, cd, cmp = -coeffs["Cl"], coeffs["Cd"], coeffs["CmPitch"]
    w = min(qa.window, len(cl) // 2)
    tail, prev = cl[-w:], cl[-2 * w : -w]
    res.cl_down = float(tail.mean())
    res.cd = float(cd[-w:].mean())
    res.cm = float(cmp[-w:].mean())
    res.efficiency = res.cl_down / res.cd if res.cd > 0 else float("nan")
    res.cl_rel_std = float(tail.std() / max(abs(tail.mean()), 1e-9))
    res.cl_rel_drift = float(abs(tail.mean() - prev.mean()) / max(abs(tail.mean()), 1e-9))
    res.iterations = int(coeffs["Time"][-1])
    yp = parse.parse_yplus(case_dir / log).get("wing", {})
    res.yplus_wing_mean, res.yplus_wing_max = yp.get("mean", np.nan), yp.get("max", np.nan)
    converged = res.cl_rel_std < qa.max_rel_std and res.cl_rel_drift < qa.max_rel_drift
    res.status = "ok" if (converged and res.mesh_ok) else ("unconverged" if not converged else "mesh_failed")


RETRY_SCRIPT = "simpleFoam > log.simpleFoam.retry 2>&1 && postProcess -func writeCellCentres -latestTime > log.postProcess 2>&1"


def _retry(case_dir: Path, res: CaseResult, flow: FlowSpec, runner: FoamRunner, qa: QATolerances) -> CaseResult:
    """Recovery for a numerical limit cycle: restart from the last state with
    plain SIMPLE and stronger under-relaxation, then judge convergence again."""
    write_fv_solution(case_dir, flow, robust=True)
    extend_run(case_dir, res.iterations + flow.end_iter)
    proc = runner.run(case_dir, RETRY_SCRIPT)
    res.extra["retried"] = 1.0
    if proc.returncode != 0:
        res.message = "retry failed: " + (proc.stderr or proc.stdout)[-300:]
        return res
    first = res.cl_rel_std
    _assess(case_dir, res, qa, "log.simpleFoam.retry")
    res.extra["cl_rel_std_before_retry"] = first
    return res


def retry_case(case_dir: Path, flow: FlowSpec | None = None, runner: FoamRunner | None = None, keep_fields: bool = True) -> CaseResult:
    """Apply the recovery step to an existing unconverged case directory."""
    flow = flow or FlowSpec()
    runner = runner or FoamRunner()
    t0 = time.time()
    res = CaseResult(**json.loads((case_dir / "result.json").read_text()))
    if res.status != "unconverged":
        return res
    elapsed = res.wall_time_s
    res = _retry(case_dir, res, flow, runner, QATolerances())
    design = WingDesign(**res.design)
    if keep_fields and res.status in ("ok", "unconverged"):
        try:
            _extract_fields(case_dir, design, place_wing(design), flow)
        except Exception as exc:
            res.message = f"field extraction failed: {exc}"
    res = _finish(res, case_dir, t0)
    res.wall_time_s += elapsed
    (case_dir / "result.json").write_text(json.dumps(asdict(res), indent=2, default=float))
    return res


def _finish(res: CaseResult, case_dir: Path, t0: float) -> CaseResult:
    res.wall_time_s = round(time.time() - t0, 1)
    (case_dir / "result.json").write_text(json.dumps(asdict(res), indent=2, default=float))
    return res


def surface_cp(design: WingDesign, wing: PlacedWing, xyz: np.ndarray, p: np.ndarray, u_inf: float) -> dict[str, np.ndarray]:
    """Cp at the fixed x/c stations on the suction (ground-facing) and pressure sides."""
    cp = p / (0.5 * u_inf**2)
    local = wing.to_local(xyz[:, :2])
    # Camber line of the upright section: mid-point of upper/lower at equal index.
    upper, lower = naca4(design, 400)
    camber_x = 0.5 * (upper[:, 0] + lower[:, 0])
    camber_y = 0.5 * (upper[:, 1] + lower[:, 1])
    yc = np.interp(local[:, 0], camber_x, camber_y)
    suction = local[:, 1] >= yc  # upright upper surface faces the ground once inverted
    st = cp_stations()
    out = {"x_c": st}
    for name, mask in (("cp_suction", suction), ("cp_pressure", ~suction)):
        order = np.argsort(local[mask, 0])
        out[name] = np.interp(st, local[mask, 0][order], cp[mask][order])
    return out


def _extract_fields(case_dir: Path, design: WingDesign, wing: PlacedWing, flow: FlowSpec) -> None:
    c, u_inf = wing.chord, flow.u_inf
    surf_dir = sorted((case_dir / "postProcessing" / "wingSurface").iterdir(), key=lambda p: float(p.name))[-1]
    xyz, p_wall = parse.read_raw_surface(surf_dir / "p_wing.raw")
    cp = surface_cp(design, wing, xyz, p_wall, u_inf)
    gxyz, p_ground = parse.read_raw_surface(surf_dir / "p_ground.raw")
    order = np.argsort(gxyz[:, 0])
    ground_x = np.linspace(-1.0, 3.0, 161)
    ground_cp = np.interp(ground_x * c + wing.le[0], gxyz[order, 0], p_ground[order] / (0.5 * u_inf**2))

    t = parse.latest_time(case_dir)
    centres = parse.read_internal_field(case_dir / t / "C")[:, :2]
    vel = parse.read_internal_field(case_dir / t / "U")[:, :2] / u_inf
    pres = parse.read_internal_field(case_dir / t / "p") / (0.5 * u_inf**2)
    nut = parse.read_internal_field(case_dir / t / "nut") / flow.nu
    xy = (centres - np.array([wing.le[0], 0.0])) / c  # chord units, origin at LE / ground
    values = np.column_stack([vel, pres])

    # Gridded window (for image-style models and plots).
    gx = np.linspace(*GRID_X, GRID_SHAPE[1])
    gy = np.linspace(*GRID_Y, GRID_SHAPE[0])
    mx, my = np.meshgrid(gx, gy)
    pts = np.column_stack([mx.ravel(), my.ravel()])
    interp = LinearNDInterpolator(xy, values, fill_value=np.nan)
    grid = interp(pts).reshape(*GRID_SHAPE, 3)
    outline = (wing.loop() - np.array([wing.le[0], 0.0])) / c
    inside = PolyPath(outline).contains_points(pts).reshape(GRID_SHAPE)
    grid[inside] = np.nan

    # Cell graph inside a window around the wing (for MeshGraphNet).
    keep = (xy[:, 0] > GRAPH_X[0]) & (xy[:, 0] < GRAPH_X[1]) & (xy[:, 1] < GRAPH_Y[1])
    idx = np.full(len(xy), -1, dtype=np.int64)
    idx[keep] = np.arange(int(keep.sum()))
    edges = parse.cell_adjacency(case_dir)
    e = idx[edges]
    e = e[(e >= 0).all(axis=1)]
    surf_dense = np.vstack([_densify(wing.suction), _densify(wing.pressure)])
    d_wall, _ = cKDTree((surf_dense - np.array([wing.le[0], 0.0])) / c).query(xy[keep])

    np.savez_compressed(
        case_dir / "fields.npz",
        x_c=cp["x_c"],
        cp_suction=cp["cp_suction"],
        cp_pressure=cp["cp_pressure"],
        ground_x=ground_x,
        ground_cp=ground_cp,
        grid=grid.astype(np.float32),
        grid_mask=inside,
        graph_pos=xy[keep].astype(np.float32),
        graph_wall_dist=d_wall.astype(np.float32),
        graph_edges=e.astype(np.int32),
        graph_y=values[keep].astype(np.float32),
        graph_nut=nut[keep].astype(np.float32),
    )


def _densify(curve: np.ndarray, n: int = 2000) -> np.ndarray:
    """Resample a polyline uniformly in arc length."""
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
    t = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(t, s, curve[:, 0]), np.interp(t, s, curve[:, 1])])


def run_batch(
    designs: Iterable[tuple[str, WingDesign]],
    root: Path,
    *,
    workers: int = 8,
    mesh: MeshSpec | None = None,
    flow: FlowSpec | None = None,
    runner: FoamRunner | None = None,
    keep_fields: bool = True,
    skip_existing: bool = True,
    on_result: Callable[[CaseResult], None] | None = None,
) -> list[CaseResult]:
    """Run many cases concurrently (one OpenFOAM process per case)."""
    jobs = list(designs)
    results: list[CaseResult] = []
    todo = []
    for case_id, design in jobs:
        rj = root / case_id / "result.json"
        if skip_existing and rj.exists():
            results.append(CaseResult(**json.loads(rj.read_text())))
        else:
            todo.append((case_id, design))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_case, d, root / cid, mesh=mesh, flow=flow, runner=runner, keep_fields=keep_fields): cid for cid, d in todo
        }
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            if on_result:
                on_result(r)
    return sorted(results, key=lambda r: r.case_id)
