"""Parsers for OpenFOAM outputs: force histories, residuals, logs, fields, mesh."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def read_table(path: Path) -> dict[str, Array]:
    """Read a functionObject ``.dat`` file into columns keyed by header name."""
    header: list[str] = []
    rows: list[list[float]] = []
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            names = line.lstrip("#").split()
            if names and names[0] == "Time":
                header = names
            continue
        parts = line.split()
        if not parts:
            continue
        row = []
        for x in parts:
            try:
                row.append(float(x))
            except ValueError:
                row.append(np.nan)  # solver names, booleans
        rows.append(row)
    if not header:
        raise ValueError(f"no header in {path}")
    width = len(header)
    data = np.array([r[:width] + [np.nan] * (width - len(r)) for r in rows], dtype=float)
    return {name: data[:, i] for i, name in enumerate(header)}


def latest_dat(case: Path, function: str, name: str) -> Path:
    files = sorted((case / "postProcessing" / function).glob(f"*/{name}"), key=lambda p: float(p.parent.name))
    if not files:
        raise FileNotFoundError(f"{function}/{name} not found in {case}")
    return files[-1]


def latest_time(case: Path) -> str:
    times = []
    for p in case.iterdir():
        try:
            if p.is_dir() and float(p.name) > 0:
                times.append(p.name)
        except ValueError:
            continue
    if not times:
        raise FileNotFoundError(f"no solution time directories in {case}")
    return max(times, key=float)


@dataclass(frozen=True)
class CheckMesh:
    ok: bool
    cells: int
    max_non_ortho: float
    max_skewness: float
    max_aspect: float


def parse_checkmesh(log: Path) -> CheckMesh:
    text = log.read_text()

    def grab(pattern: str, default: float = np.nan) -> float:
        m = re.search(pattern, text)
        return float(m.group(1)) if m else default

    return CheckMesh(
        ok="Mesh OK." in text,
        cells=int(grab(r"cells:\s+(\d+)", 0)),
        max_non_ortho=grab(r"Mesh non-orthogonality Max: ([\d.eE+-]+)"),
        max_skewness=grab(r"Max skewness = ([\d.eE+-]+)"),
        max_aspect=grab(r"Max aspect ratio = ([\d.eE+-]+)"),
    )


def parse_yplus(log: Path) -> dict[str, dict[str, float]]:
    """Last reported y+ statistics per patch from the solver log."""
    out: dict[str, dict[str, float]] = {}
    pat = re.compile(r"patch (\w+) y\+ : min = ([\d.eE+-]+), max = ([\d.eE+-]+), average = ([\d.eE+-]+)")
    for m in pat.finditer(log.read_text()):
        out[m.group(1)] = {"min": float(m.group(2)), "max": float(m.group(3)), "mean": float(m.group(4))}
    return out


def read_raw_surface(path: Path) -> tuple[Array, Array]:
    """Read a ``raw`` surface sample: returns (xyz [N,3], values [N] or [N,k])."""
    data = np.loadtxt(path, comments="#", ndmin=2)
    return data[:, :3], data[:, 3:].squeeze(-1) if data.shape[1] == 4 else data[:, 3:]


_LIST_RE = re.compile(r"(\d+)\s*\(")


def _parse_list(text: str, start: int, ncomp: int) -> tuple[Array, int]:
    m = _LIST_RE.search(text, start)
    if m is None:
        raise ValueError("list not found")
    n = int(m.group(1))
    body_start = m.end()
    end = _matching_paren(text, m.end() - 1)
    body = text[body_start:end].replace("(", " ").replace(")", " ")
    arr = np.array(body.split(), dtype=float)
    if arr.size != n * ncomp:
        raise ValueError(f"expected {n}x{ncomp} values, got {arr.size}")
    return (arr.reshape(n, ncomp) if ncomp > 1 else arr), end


def _matching_paren(text: str, open_idx: int) -> int:
    depth = 0
    for i in range(open_idx, len(text)):
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("unbalanced parentheses")


def read_internal_field(path: Path) -> Array:
    """Internal field values of an ASCII volScalarField/volVectorField."""
    text = path.read_text()
    m = re.search(r"internalField\s+(uniform|nonuniform)", text)
    if m is None:
        raise ValueError(f"no internalField in {path}")
    if m.group(1) == "uniform":
        raise ValueError(f"{path} has a uniform internal field")
    ncomp = 3 if "List<vector>" in text[m.end() : m.end() + 40] else 1
    arr, _ = _parse_list(text, m.end(), ncomp)
    return arr


def read_label_list(path: Path) -> NDArray[np.int64]:
    text = path.read_text()
    start = text.find("}") + 1  # skip FoamFile header
    m = _LIST_RE.search(text, start)
    if m is None:
        raise ValueError(f"no list in {path}")
    n = int(m.group(1))
    end = _matching_paren(text, m.end() - 1)
    arr = np.array(text[m.end() : end].split(), dtype=np.int64)
    if arr.size != n:
        raise ValueError(f"{path}: expected {n} labels, got {arr.size}")
    return arr


def cell_adjacency(case: Path) -> NDArray[np.int64]:
    """Undirected cell-cell edges [E, 2] from internal faces (owner/neighbour)."""
    mesh = case / "constant" / "polyMesh"
    owner = read_label_list(mesh / "owner")
    neigh = read_label_list(mesh / "neighbour")
    return np.column_stack([owner[: neigh.size], neigh])
