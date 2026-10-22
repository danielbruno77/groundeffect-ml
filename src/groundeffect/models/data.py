"""Dataset loading, feature scaling and POD compression of surface pressure."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from groundeffect.doe import PARAMS, DesignSpace

SCALARS = ("cl_down", "cd", "cm")


@dataclass
class Split:
    case_id: np.ndarray
    x: np.ndarray  # physical design variables [n, 4]
    y: np.ndarray  # scalar targets [n, 3]
    cp: np.ndarray  # surface Cp [n, 160] (suction then pressure side)


@dataclass
class Dataset:
    train: Split
    test: Split
    x_c: np.ndarray
    cases: pd.DataFrame

    @property
    def n_stations(self) -> int:
        return len(self.x_c)


def load_dataset(path: Path) -> Dataset:
    cases = pd.read_csv(path / "cases.csv")
    surf = np.load(path / "surface.npz")
    ids = list(surf["case_id"])
    cp = np.hstack([surf["cp_suction"], surf["cp_pressure"]])
    ok = cases[(cases["status"] == "ok") & cases["case_id"].isin(ids)].set_index("case_id")
    row = {cid: i for i, cid in enumerate(ids)}

    def split(name: str) -> Split:
        sub = ok[ok["split"] == name]
        idx = [row[c] for c in sub.index]
        return Split(
            case_id=sub.index.to_numpy(),
            x=sub[list(PARAMS)].to_numpy(float),
            y=sub[list(SCALARS)].to_numpy(float),
            cp=cp[idx],
        )

    return Dataset(train=split("train"), test=split("test"), x_c=surf["x_c"], cases=cases)


@dataclass
class Scaler:
    """Inputs go to the unit cube of the design space (log ride height)."""

    space: DesignSpace

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self.space.to_unit(x) * 2.0 - 1.0


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, y: np.ndarray) -> Standardizer:
        return cls(y.mean(0), y.std(0) + 1e-12)

    def transform(self, y: np.ndarray) -> np.ndarray:
        return (y - self.mean) / self.std

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return z * self.std + self.mean


@dataclass
class POD:
    """Proper orthogonal decomposition (via SVD) of a set of snapshots."""

    mean: np.ndarray
    modes: np.ndarray  # [k, n_features]
    energy: np.ndarray  # cumulative energy fraction per mode

    @classmethod
    def fit(cls, snapshots: np.ndarray, energy: float = 0.9995, max_modes: int = 24) -> POD:
        mean = snapshots.mean(0)
        _, s, vt = np.linalg.svd(snapshots - mean, full_matrices=False)
        cum = np.cumsum(s**2) / np.sum(s**2)
        k = min(max_modes, int(np.searchsorted(cum, energy)) + 1)
        return cls(mean=mean, modes=vt[:k], energy=cum[:k])

    @property
    def k(self) -> int:
        return len(self.modes)

    def encode(self, snapshots: np.ndarray) -> np.ndarray:
        return (snapshots - self.mean) @ self.modes.T

    def decode(self, coeffs: np.ndarray) -> np.ndarray:
        return coeffs @ self.modes + self.mean
