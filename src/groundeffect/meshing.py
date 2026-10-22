"""Automated 2D hybrid meshing with gmsh.

Builds a prism-layer + triangle mesh around the wing inside a rectangular
tunnel with a moving ground, then extrudes it one cell in z so OpenFOAM can run
it as a 2D case (``gmshToFoam`` + ``empty`` front/back patches).

All sizes scale with a single ``refinement`` factor, so a family of
systematically refined meshes can be produced for a grid convergence study.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from groundeffect.geometry import PlacedWing


@dataclass(frozen=True)
class MeshSpec:
    """Mesh controls, lengths as fractions of chord unless stated."""

    refinement: float = 1.0  # >1 is finer; all sizes divide by this
    upstream: float = 4.0
    downstream: float = 8.0
    height: float = 4.0
    surface_size: float = 0.008
    le_te_size: float = 0.002
    first_layer: float = 4.0e-5  # absolute first-cell height [m] at refinement 1
    growth: float = 1.15
    bl_thickness: float = 0.035  # must contain the physical BL (~3-4% c at the TE, Re 5e5)
    near_size: float = 0.006  # matches the last prism layer: no size jump at the BL edge
    near_wake_size: float = 0.008
    wake_size: float = 0.025
    gap_size: float = 0.01
    far_size: float = 0.4
    span: float = 0.01  # extrusion depth

    def scaled(self, value: float) -> float:
        return value / self.refinement


@dataclass(frozen=True)
class MeshStats:
    n_nodes_2d: int
    n_cells: int
    msh_path: Path


PATCHES = ("inlet", "outlet", "top", "ground", "wing", "frontAndBack")


def build_mesh(wing: PlacedWing, out: Path, spec: MeshSpec | None = None) -> MeshStats:
    """Mesh the domain around ``wing`` and write a gmsh v2.2 file to ``out``."""
    import gmsh  # optional dependency (the `cfd` extra): only meshing needs it

    spec = spec or MeshSpec()
    c = wing.chord
    s = spec.scaled
    out.parent.mkdir(parents=True, exist_ok=True)

    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.model.add("wing_in_ground_effect")
        geo = gmsh.model.geo

        loop = wing.loop()
        i_le = len(wing.pressure) - 1  # index of the LE inside the loop
        tags = [geo.addPoint(float(x), float(y), 0.0) for x, y in loop]
        pressure_curve = geo.addSpline(tags[: i_le + 1])
        suction_curve = geo.addSpline([*tags[i_le:], tags[0]])
        wing_loop = geo.addCurveLoop([pressure_curve, suction_curve])

        x0, x1 = wing.le[0] - spec.upstream * c, wing.te[0] + spec.downstream * c
        y1 = spec.height * c
        far = s(spec.far_size) * c
        corners = [geo.addPoint(x0, 0, 0, far), geo.addPoint(x1, 0, 0, far)]
        corners += [geo.addPoint(x1, y1, 0, far), geo.addPoint(x0, y1, 0, far)]
        ground = geo.addLine(corners[0], corners[1])
        outlet = geo.addLine(corners[1], corners[2])
        top = geo.addLine(corners[2], corners[3])
        inlet = geo.addLine(corners[3], corners[0])
        outer = geo.addCurveLoop([ground, outlet, top, inlet])
        surface = geo.addPlaneSurface([outer, wing_loop])
        # Surface node distribution, bunched at LE and TE. Set through the geo
        # kernel: constraints set on gmsh.model.mesh are lost on re-synchronise.
        n_surf = max(40, int(round(1.0 / s(spec.surface_size))))
        for curve in (pressure_curve, suction_curve):
            geo.mesh.setTransfiniteCurve(curve, n_surf, "Bump", spec.le_te_size / spec.surface_size)
        geo.synchronize()

        field = gmsh.model.mesh.field
        bl = field.add("BoundaryLayer")
        field.setNumbers(bl, "CurvesList", [pressure_curve, suction_curve])
        field.setNumbers(bl, "FanPointsList", [tags[0]])
        field.setNumber(bl, "Size", s(spec.first_layer))
        field.setNumber(bl, "Ratio", spec.growth)
        h_gap = float(min(wing.suction[:, 1].min(), wing.pressure[:, 1].min()))
        field.setNumber(bl, "Thickness", min(spec.bl_thickness * c, 0.4 * h_gap))
        field.setNumber(bl, "Quads", 1)
        field.setAsBoundaryLayer(bl)

        dist = field.add("Distance")
        field.setNumbers(dist, "CurvesList", [pressure_curve, suction_curve])
        field.setNumber(dist, "Sampling", 400)
        near = field.add("Threshold")
        field.setNumber(near, "InField", dist)
        field.setNumber(near, "SizeMin", s(spec.near_size) * c)
        field.setNumber(near, "SizeMax", far)
        field.setNumber(near, "DistMin", 0.04 * c)
        field.setNumber(near, "DistMax", 3.0 * c)

        y_top_wing = max(wing.pressure[:, 1].max(), wing.suction[:, 1].max())
        wake = field.add("Box")
        field.setNumber(wake, "VIn", s(spec.wake_size) * c)
        field.setNumber(wake, "VOut", far)
        field.setNumber(wake, "XMin", wing.le[0] - 0.2 * c)
        field.setNumber(wake, "XMax", wing.te[0] + 2.5 * c)
        field.setNumber(wake, "YMin", 0.0)
        field.setNumber(wake, "YMax", y_top_wing + 0.5 * c)
        field.setNumber(wake, "Thickness", 0.5 * c)

        # Near wake: the shear layer leaving the TE.
        near_wake = field.add("Box")
        field.setNumber(near_wake, "VIn", s(spec.near_wake_size) * c)
        field.setNumber(near_wake, "VOut", far)
        field.setNumber(near_wake, "XMin", wing.te[0] - 0.05 * c)
        field.setNumber(near_wake, "XMax", wing.te[0] + 1.0 * c)
        field.setNumber(near_wake, "YMin", max(0.0, wing.te[1] - 0.2 * c))
        field.setNumber(near_wake, "YMax", wing.te[1] + 0.15 * c)
        field.setNumber(near_wake, "Thickness", 0.2 * c)

        # Resolve the ground/wing gap: size tied to the actual gap height.
        h = h_gap
        gap = field.add("Box")
        field.setNumber(gap, "VIn", min(s(spec.gap_size) * c, s(h / 8.0)))
        field.setNumber(gap, "VOut", far)
        field.setNumber(gap, "XMin", wing.le[0] - 0.3 * c)
        field.setNumber(gap, "XMax", wing.te[0] + 1.0 * c)
        field.setNumber(gap, "YMin", 0.0)
        field.setNumber(gap, "YMax", h + 0.05 * c)
        field.setNumber(gap, "Thickness", 0.3 * c)

        bg = field.add("Min")
        field.setNumbers(bg, "FieldsList", [near, near_wake, wake, gap])
        field.setAsBackgroundMesh(bg)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)

        extruded = geo.extrude([(2, surface)], 0, 0, spec.span, numElements=[1], recombine=True)
        geo.synchronize()

        front_back = [surface]
        groups: dict[str, list[int]] = {name: [] for name in PATCHES}
        volume = None
        for dim, tag in extruded:
            if dim == 3:
                volume = tag
            elif dim == 2:
                groups[_classify(tag, x0, x1, y1)].append(tag)
        groups["frontAndBack"] += front_back
        if volume is None:
            raise RuntimeError("extrusion did not produce a volume")
        for name, surfaces in groups.items():
            if not surfaces:
                raise RuntimeError(f"no surfaces classified as {name}")
            gmsh.model.addPhysicalGroup(2, surfaces, name=name)
        gmsh.model.addPhysicalGroup(3, [volume], name="fluid")

        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.model.mesh.generate(3)
        gmsh.write(str(out))

        n_nodes_2d = len(gmsh.model.mesh.getNodes(2, surface, includeBoundary=True)[0])
        types, elems, _ = gmsh.model.mesh.getElements(3, volume)
        n_cells = int(sum(len(e) for e in elems))
        del types
        return MeshStats(n_nodes_2d=n_nodes_2d, n_cells=n_cells, msh_path=out)
    finally:
        gmsh.finalize()


def _classify(tag: int, x0: float, x1: float, y1: float) -> str:
    """Name an extruded lateral surface from its bounding box."""
    import gmsh

    if tag < 0:
        tag = -tag
    bx0, by0, bz0, bx1, by1, bz1 = gmsh.model.getBoundingBox(2, tag)
    tol = 1e-6 * max(abs(x1 - x0), 1.0)
    if abs(bz1 - bz0) < tol:
        return "frontAndBack"
    if abs(bx1 - bx0) < tol and abs(bx0 - x0) < tol:
        return "inlet"
    if abs(bx1 - bx0) < tol and abs(bx0 - x1) < tol:
        return "outlet"
    if abs(by1 - by0) < tol and abs(by0) < tol:
        return "ground"
    if abs(by1 - by0) < tol and abs(by0 - y1) < tol:
        return "top"
    return "wing"
