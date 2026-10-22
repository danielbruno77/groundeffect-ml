"""Solution verification: grid convergence index (Celik et al., JFE 2008).

The GCI gives the numerical uncertainty of the CFD itself. It sets the floor
for surrogate accuracy targets: a surrogate error well below the
discretisation uncertainty of its training data is not a meaningful goal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GCIResult:
    phi: tuple[float, float, float]  # fine, medium, coarse
    h: tuple[float, float, float]
    order: float  # apparent order p
    phi_extrapolated: float
    rel_error_fine: float  # e_a21
    gci_fine: float  # relative, as a fraction
    gci_medium: float
    asymptotic_ratio: float  # ~1 when in the asymptotic range
    oscillatory: bool


def representative_h(n_cells: int, area: float = 1.0) -> float:
    """Representative cell size for a 2D mesh: (A / N)^(1/2)."""
    return float(np.sqrt(area / n_cells))


def gci(phi: tuple[float, float, float], n_cells: tuple[int, int, int], fs: float = 1.25) -> GCIResult:
    """Three-grid GCI. ``phi`` and ``n_cells`` are ordered fine, medium, coarse."""
    f1, f2, f3 = phi
    h1, h2, h3 = (representative_h(n) for n in n_cells)
    r21, r32 = h2 / h1, h3 / h2
    e21, e32 = f2 - f1, f3 - f2
    if e21 == 0 or e32 == 0:
        raise ValueError("solutions are identical on two grids; GCI undefined")
    s = float(np.sign(e32 / e21))
    p = 2.0
    for _ in range(200):
        q = np.log((r21**p - s) / (r32**p - s))
        p_new = abs(np.log(abs(e32 / e21)) + q) / np.log(r21)
        if abs(p_new - p) < 1e-10:
            break
        p = p_new
    p = float(np.clip(p, 0.5, 6.0))  # guard against wild values on noisy data
    f_ext = (r21**p * f1 - f2) / (r21**p - 1.0)
    ea21 = abs((f1 - f2) / f1)
    ea32 = abs((f2 - f3) / f2)
    gci21 = fs * ea21 / (r21**p - 1.0)
    gci32 = fs * ea32 / (r32**p - 1.0)
    return GCIResult(
        phi=(f1, f2, f3),
        h=(h1, h2, h3),
        order=p,
        phi_extrapolated=float(f_ext),
        rel_error_fine=float(ea21),
        gci_fine=float(gci21),
        gci_medium=float(gci32),
        asymptotic_ratio=float(gci32 / (r21**p * gci21)) if gci21 > 0 else float("nan"),
        oscillatory=s < 0,
    )
