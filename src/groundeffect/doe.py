"""Design of experiments over the wing design space."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

from groundeffect.geometry import WingDesign

PARAMS = ("h_c", "alpha_deg", "camber", "thickness")


@dataclass(frozen=True)
class DesignSpace:
    h_c: tuple[float, float] = (0.08, 0.60)
    alpha_deg: tuple[float, float] = (0.0, 8.0)
    camber: tuple[float, float] = (0.02, 0.06)
    thickness: tuple[float, float] = (0.09, 0.15)
    log_ride_height: bool = True  # ground effect is strongest (and most non-linear) at low h

    def bounds(self) -> np.ndarray:
        return np.array([self.h_c, self.alpha_deg, self.camber, self.thickness], dtype=float)

    def from_unit(self, u: np.ndarray) -> np.ndarray:
        """Map points in [0, 1]^d to physical design variables."""
        u = np.atleast_2d(u)
        b = self.bounds()
        x = b[:, 0] + u * (b[:, 1] - b[:, 0])
        if self.log_ride_height:
            lo, hi = np.log(b[0])
            x[:, 0] = np.exp(lo + u[:, 0] * (hi - lo))
        return x

    def to_unit(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        b = self.bounds()
        u = (x - b[:, 0]) / (b[:, 1] - b[:, 0])
        if self.log_ride_height:
            lo, hi = np.log(b[0])
            u[:, 0] = (np.log(x[:, 0]) - lo) / (hi - lo)
        return u

    def designs(self, x: np.ndarray) -> list[WingDesign]:
        return [WingDesign(*map(float, row)) for row in np.atleast_2d(x)]


def sobol(space: DesignSpace, n: int, seed: int = 0) -> np.ndarray:
    """Scrambled Sobol points (n rounded up to a power of two is best practice)."""
    sampler = qmc.Sobol(d=len(PARAMS), scramble=True, seed=seed)
    return space.from_unit(sampler.random(n))


def latin_hypercube(space: DesignSpace, n: int, seed: int = 1) -> np.ndarray:
    sampler = qmc.LatinHypercube(d=len(PARAMS), seed=seed)
    return space.from_unit(sampler.random(n))
