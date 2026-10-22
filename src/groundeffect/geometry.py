"""Parametric inverted wing in ground effect.

The wing is a NACA 4-digit section, mirrored so the suction side faces the
ground (a single-element front wing). Incidence is nose-down about the quarter
chord, and ride height ``h`` is the gap between the ground and the lowest point
of the section, following Zerihan & Zhang (J. Aircraft, 2000).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class WingDesign:
    """Design variables. Lengths are normalised by chord."""

    h_c: float  # ride height / chord
    alpha_deg: float  # nose-down incidence [deg]
    camber: float  # NACA max camber m (fraction of chord)
    thickness: float  # NACA max thickness t (fraction of chord)
    camber_pos: float = 0.4  # NACA camber position p

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    @property
    def naca_code(self) -> str:
        return f"{round(self.camber * 100):d}{round(self.camber_pos * 10):d}{round(self.thickness * 100):02d}"


def naca4(design: WingDesign, n_side: int = 160) -> tuple[Array, Array]:
    """Return (upper, lower) surfaces of the upright section, LE to TE, chord 1.

    Uses cosine spacing and the closed-trailing-edge thickness coefficient so the
    two surfaces meet at a single TE point.
    """
    m, p, t = design.camber, design.camber_pos, design.thickness
    beta = np.linspace(0.0, np.pi, n_side)
    x = 0.5 * (1.0 - np.cos(beta))
    yt = 5.0 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
    yc = np.zeros_like(x)
    dyc = np.zeros_like(x)
    if m > 0:
        fwd = x < p
        yc[fwd] = m / p**2 * (2 * p * x[fwd] - x[fwd] ** 2)
        dyc[fwd] = 2 * m / p**2 * (p - x[fwd])
        aft = ~fwd
        yc[aft] = m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x[aft] - x[aft] ** 2)
        dyc[aft] = 2 * m / (1 - p) ** 2 * (p - x[aft])
    theta = np.arctan(dyc)
    upper = np.column_stack([x - yt * np.sin(theta), yc + yt * np.cos(theta)])
    lower = np.column_stack([x + yt * np.sin(theta), yc - yt * np.cos(theta)])
    upper[-1] = lower[-1] = (1.0, 0.0)
    return upper, lower


def _rotate(xy: Array, angle_rad: float, origin: tuple[float, float]) -> Array:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    d = xy - np.asarray(origin)
    return np.column_stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]]) + np.asarray(origin)


@dataclass(frozen=True)
class PlacedWing:
    """Wing in the ground frame (ground at y = 0, flow in +x), dimensional."""

    suction: Array  # surface facing the ground, LE -> TE
    pressure: Array  # surface facing up, LE -> TE
    chord: float
    quarter_chord: tuple[float, float]
    alpha_rad: float

    @property
    def le(self) -> Array:
        return self.suction[0]

    @property
    def te(self) -> Array:
        return self.suction[-1]

    def loop(self) -> Array:
        """Closed loop TE -> pressure side -> LE -> suction side, without repeating TE."""
        return np.vstack([self.pressure[::-1][:-1], self.suction[:-1]])

    def to_local(self, xy: Array) -> Array:
        """Map ground-frame points back to the chord-aligned, upright unit-chord frame."""
        pts = _rotate(np.asarray(xy, dtype=float), self.alpha_rad, self.quarter_chord)
        pts = (pts - np.asarray(self.quarter_chord)) / self.chord
        pts[:, 1] *= -1.0
        pts[:, 0] += 0.25
        return pts


def place_wing(design: WingDesign, chord: float = 0.3, x_le: float = 0.0, n_side: int = 160) -> PlacedWing:
    """Invert, rotate nose-down by alpha about c/4 and set the ride height."""
    upper, lower = naca4(design, n_side)
    # Mirror in y: the upright suction (upper) side now faces the ground.
    suction = np.column_stack([upper[:, 0], -upper[:, 1]]) * chord
    pressure = np.column_stack([lower[:, 0], -lower[:, 1]]) * chord
    qc = (0.25 * chord, 0.0)
    a = np.deg2rad(design.alpha_deg)
    # Positive rotation (counter-clockwise) lifts the TE: nose-down incidence.
    suction = _rotate(suction, a, qc)
    pressure = _rotate(pressure, a, qc)
    dy = design.h_c * chord - min(suction[:, 1].min(), pressure[:, 1].min())
    shift = np.array([x_le, dy])
    return PlacedWing(
        suction=suction + shift,
        pressure=pressure + shift,
        chord=chord,
        quarter_chord=(qc[0] + x_le, qc[1] + dy),
        alpha_rad=-a,
    )


def cp_stations(n: int = 80) -> Array:
    """Fixed x/c stations (cosine clustered) at which surface Cp is stored."""
    beta = np.linspace(0.0, np.pi, n + 2)[1:-1]
    return 0.5 * (1.0 - np.cos(beta))
