import numpy as np
import pytest
import torch

from groundeffect.doe import DesignSpace, sobol
from groundeffect.models.data import POD
from groundeffect.models.ensemble import EnsembleSurrogate, TrainConfig, export_program
from groundeffect.models.evaluate import check_gates, delta_metrics, scalar_metrics
from groundeffect.models.gp import GPSurrogate, active_learning_curve


def toy(x):
    """Smooth stand-in for the CFD map with a ground-effect-like 1/h term."""
    h, a, m, t = x.T
    cl = 0.3 + 0.09 * a + 10 * m + 0.05 / (h + 0.05)
    cd = 0.01 + 0.02 * cl**2 + 0.05 * t
    cm = 0.1 + 0.2 * m - 0.005 * a
    xs = np.linspace(0, 1, 40)
    cp = -cl[:, None] * np.exp(-((xs - 0.3) ** 2) / 0.05) + 0.2 * t[:, None]
    return np.column_stack([cl, cd, cm]), np.hstack([cp, 0.5 * cp])


@pytest.fixture(scope="module")
def data():
    space = DesignSpace()
    x = sobol(space, 64, seed=3)
    xt = sobol(space, 16, seed=9)
    y, cp = toy(x)
    yt, cpt = toy(xt)
    return x, y, cp, xt, yt, cpt


@pytest.fixture(scope="module")
def surrogate(data):
    x, y, cp, *_ = data
    return EnsembleSurrogate(config=TrainConfig(members=3, epochs=600, width=64)).fit(x, y, cp)


def test_pod_reconstructs_low_rank_data():
    rng = np.random.default_rng(0)
    snaps = rng.normal(size=(50, 3)) @ rng.normal(size=(3, 80))
    pod = POD.fit(snaps, energy=0.999999)
    assert pod.k == 3
    assert np.allclose(pod.decode(pod.encode(snaps)), snaps, atol=1e-8)


def test_ensemble_learns_toy_map(surrogate, data):
    *_, xt, yt, cpt = data
    p = surrogate.predict(xt)
    assert scalar_metrics(yt[:, 0], p["scalars"][:, 0])["r2"] > 0.97
    assert np.all(p["scalars_std"] > 0)
    assert np.sqrt(np.mean((p["cp"] - cpt) ** 2)) < 0.1


def test_save_load_roundtrip(surrogate, data, tmp_path):
    *_, xt, _, _ = data
    surrogate.save(tmp_path / "m")
    again = EnsembleSurrogate.load(tmp_path / "m")
    a, b = surrogate.predict(xt), again.predict(xt)
    for k in a:
        assert np.allclose(a[k], b[k], atol=1e-6)


def test_exported_program_matches_python(surrogate, data, tmp_path):
    *_, xt, _, _ = data
    path = export_program(surrogate, tmp_path / "m.pt2")
    prog = torch.export.load(str(path)).module()
    with torch.no_grad():
        out = prog(torch.tensor(xt, dtype=torch.float32)).numpy()
    ref = surrogate.predict(xt)
    assert np.allclose(out[:, :3], ref["scalars"], atol=1e-4)
    assert np.allclose(out[:, 3:6], ref["scalars_std"], atol=1e-4)
    n = ref["cp"].shape[1]
    assert np.allclose(out[:, 6 : 6 + n], ref["cp"], atol=1e-4)
    # dynamic batch dimension
    assert prog(torch.tensor(xt[:1], dtype=torch.float32)).shape[0] == 1


def test_gp_interpolates_and_reports_uncertainty(data):
    x, y, _, xt, yt, _ = data
    gp = GPSurrogate().fit(x, y)
    mu, sd = gp.predict(xt)
    assert scalar_metrics(yt[:, 0], mu[:, 0])["r2"] > 0.99
    _, sd_train = gp.predict(x[:5])
    assert sd_train.mean() < sd.mean()  # less uncertain at training points


def test_active_learning_curve_shape(data):
    x, y, _, xt, yt, _ = data
    curve = active_learning_curve(x, y, xt, yt, n_init=8, n_max=20, step=4, strategy="max_std")
    assert [n for n, _ in curve] == [8, 12, 16, 20]
    assert curve[-1][1] < curve[0][1]


def test_gates(tmp_path):
    gates = tmp_path / "g.yaml"
    gates.write_text("cl_down:\n  r2: {min: 0.9}\n  mape: {max: 0.05}\n")
    res = check_gates({"cl_down": {"r2": 0.95, "mape": 0.08}}, gates)
    assert [g.passed for g in res] == [True, False]


def test_delta_metrics_rank_and_threshold():
    y = np.array([1.0, 1.1, 1.2, 1.3])
    perfect = delta_metrics(y, y + 0.01)
    assert perfect["pair_sign_accuracy"] == 1.0 and perfect["spearman"] == pytest.approx(1.0)
    swapped = delta_metrics(y, np.array([1.0, 1.2, 1.1, 1.3]))
    assert swapped["pair_sign_accuracy"] == pytest.approx(5 / 6)
    # The swapped pair differs by 0.1: excluded when deltas below 0.15 do not count.
    assert delta_metrics(y, np.array([1.0, 1.2, 1.1, 1.3]), min_delta=0.15)["pair_sign_accuracy"] == 1.0
