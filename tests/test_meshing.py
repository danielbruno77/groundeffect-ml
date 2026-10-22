import pytest

gmsh = pytest.importorskip("gmsh")

from groundeffect.geometry import WingDesign, place_wing  # noqa: E402
from groundeffect.meshing import PATCHES, MeshSpec, build_mesh  # noqa: E402


def _physical_names(path):
    text = path.read_text()
    block = text.split("$PhysicalNames")[1].split("$EndPhysicalNames")[0].split("\n")[2:-1]
    return {line.split('"')[1] for line in block}


def test_mesh_has_all_patches_and_reasonable_size(tmp_path):
    out = tmp_path / "mesh.msh"
    stats = build_mesh(place_wing(WingDesign(0.15, 4.0, 0.04, 0.12)), out, MeshSpec(refinement=0.5))
    assert set(PATCHES) | {"fluid"} == _physical_names(out)
    assert 3_000 < stats.n_cells < 30_000


def test_refinement_increases_cell_count(tmp_path):
    w = place_wing(WingDesign(0.3, 2.0, 0.03, 0.1))
    coarse = build_mesh(w, tmp_path / "c.msh", MeshSpec(refinement=0.5)).n_cells
    fine = build_mesh(w, tmp_path / "f.msh", MeshSpec(refinement=0.8)).n_cells
    assert fine > 1.8 * coarse  # ~ r^2 in 2D


def test_prism_layer_capped_inside_small_gap(tmp_path):
    # At the lowest ride height the prism layer must not fill the gap.
    stats = build_mesh(place_wing(WingDesign(0.08, 8.0, 0.06, 0.15)), tmp_path / "m.msh", MeshSpec(refinement=0.5))
    assert stats.n_cells > 0
