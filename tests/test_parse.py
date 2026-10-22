import numpy as np
import pytest

from groundeffect.foam import parse


def test_force_table(data_dir):
    t = parse.read_table(data_dir / "coefficient.dat")
    assert {"Time", "Cd", "Cl", "CmPitch"} <= set(t)
    assert len(t["Time"]) > 10 and np.all(np.diff(t["Time"]) > 0)
    assert np.all(t["Cl"] < 0)  # downforce


def test_solver_info_handles_text_columns(data_dir):
    t = parse.read_table(data_dir / "solverInfo.dat")
    assert np.isfinite(t["Ux_initial"]).all()


def test_checkmesh(data_dir):
    cm = parse.parse_checkmesh(data_dir / "log.checkMesh")
    assert cm.ok and cm.cells > 1000
    assert 0 < cm.max_non_ortho < 70


def test_yplus(data_dir):
    yp = parse.parse_yplus(data_dir / "log.simpleFoam")
    assert {"wing", "ground"} <= set(yp)
    assert yp["wing"]["min"] <= yp["wing"]["mean"] <= yp["wing"]["max"]


@pytest.mark.parametrize("ncomp", [1, 3])
def test_internal_field_roundtrip(tmp_path, ncomp):
    rng = np.random.default_rng(0)
    vals = rng.normal(size=(5, ncomp)) if ncomp == 3 else rng.normal(size=5)
    kind = "vector" if ncomp == 3 else "scalar"
    body = "\n".join("(" + " ".join(f"{v:.9g}" for v in row) + ")" for row in vals) if ncomp == 3 else "\n".join(f"{v:.9g}" for v in vals)
    text = f"FoamFile {{ format ascii; }}\ninternalField   nonuniform List<{kind}> \n5\n(\n{body}\n)\n;\nboundaryField {{ wing {{ type zeroGradient; }} }}\n"
    f = tmp_path / "field"
    f.write_text(text)
    assert np.allclose(parse.read_internal_field(f), vals)


def test_label_list(tmp_path):
    f = tmp_path / "owner"
    f.write_text('FoamFile\n{\n note "nPoints:4";\n}\n\n4\n(\n0\n1\n1\n2\n)\n')
    assert parse.read_label_list(f).tolist() == [0, 1, 1, 2]
