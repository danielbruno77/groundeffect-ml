from groundeffect.foam.case import FlowSpec, write_allrun, write_case
from groundeffect.geometry import WingDesign, place_wing
from groundeffect.meshing import PATCHES


def test_case_files_cover_every_patch(tmp_path):
    write_case(tmp_path, place_wing(WingDesign(0.15, 4, 0.04, 0.12)), FlowSpec(), span=0.01)
    write_allrun(tmp_path)
    for field in ("U", "p", "k", "omega", "nut"):
        text = (tmp_path / "0" / field).read_text()
        for patch in PATCHES:
            assert f"    {patch}\n" in text, (field, patch)
    ctrl = (tmp_path / "system/controlDict").read_text()
    assert "forceCoeffs" in ctrl and "liftDir         (0 1 0)" in ctrl
    assert (tmp_path / "Allrun").stat().st_mode & 0o111


def test_freestream_turbulence():
    f = FlowSpec(u_inf=25, turbulence_intensity=0.002, nu=1.5e-5, viscosity_ratio=1)
    assert f.k() == 1.5 * (25 * 0.002) ** 2
    assert f.reynolds(0.3) == 5e5


def test_robust_profile_and_restart(tmp_path):
    from groundeffect.foam.case import extend_run, write_fv_solution

    write_case(tmp_path, place_wing(WingDesign(0.15, 4, 0.04, 0.12)), FlowSpec(end_iter=1500), span=0.01)
    write_fv_solution(tmp_path, FlowSpec(), robust=True)
    fv = (tmp_path / "system/fvSolution").read_text()
    assert "consistent      no;" in fv and "p 0.3;" in fv
    extend_run(tmp_path, 3000)
    ctrl = (tmp_path / "system/controlDict").read_text()
    assert "endTime         3000;" in ctrl and "writeInterval   3000;" in ctrl


def test_cli_imports_without_gmsh():
    # The model workflow and the serving image do not install the `cfd` extra.
    import subprocess
    import sys

    code = "import sys; sys.modules['gmsh'] = None; import groundeffect.cli, groundeffect.serve"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True).returncode == 0
