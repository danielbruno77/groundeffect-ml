import importlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

from groundeffect.doe import DesignSpace, sobol
from groundeffect.models.ensemble import EnsembleSurrogate, TrainConfig, export_program


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    art = tmp_path_factory.mktemp("artifacts")
    x = sobol(DesignSpace(), 32)
    y = np.column_stack([1 + x[:, 1] * 0.1, 0.02 + 0.01 * x[:, 3], 0.1 * x[:, 2]])
    cp = np.tile(np.linspace(-1, 0.5, 160), (32, 1)) * y[:, :1]
    sur = EnsembleSurrogate(config=TrainConfig(members=2, epochs=50, width=16)).fit(x, y, cp)
    export_program(sur, art / "ensemble.pt2")
    import os

    os.environ["GROUNDEFFECT_ARTIFACTS"] = str(art)
    import groundeffect.serve as serve

    importlib.reload(serve)
    return TestClient(serve.app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_and_extrapolation_flag(client):
    body = {
        "designs": [
            {"h_c": 0.15, "alpha_deg": 4, "camber": 0.04, "thickness": 0.12},
            {"h_c": 0.02, "alpha_deg": 12, "camber": 0.04, "thickness": 0.12},
        ]
    }
    r = client.post("/predict", json=body)
    assert r.status_code == 200
    a, b = r.json()
    assert a["extrapolation"] == [] and len(a["cp_suction"]) == 80
    assert set(b["extrapolation"]) == {"h_c", "alpha_deg"}


def test_rejects_invalid_input(client):
    r = client.post("/predict", json={"designs": [{"h_c": -1, "alpha_deg": 4, "camber": 0.04, "thickness": 0.12}]})
    assert r.status_code == 422
