"""HTTP inference service for the exported surrogate.

Loads only the ``torch.export`` program and the metrics file produced by the
training gate, so what is served is exactly what passed the gate. Requests
outside the training domain are answered but flagged as extrapolation.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

from groundeffect.doe import PARAMS, DesignSpace
from groundeffect.geometry import cp_stations

ARTIFACTS = Path(os.environ.get("GROUNDEFFECT_ARTIFACTS", "artifacts"))

app = FastAPI(title="groundeffect surrogate", version="0.1.0")


class Design(BaseModel):
    h_c: float = Field(..., gt=0, description="ride height / chord")
    alpha_deg: float = Field(..., description="nose-down incidence [deg]")
    camber: float = Field(..., ge=0, description="NACA max camber (fraction of chord)")
    thickness: float = Field(..., gt=0, description="NACA max thickness (fraction of chord)")


class PredictRequest(BaseModel):
    designs: list[Design] = Field(..., min_length=1, max_length=4096)
    include_cp: bool = True


class Prediction(BaseModel):
    cl_down: float
    cd: float
    cm: float
    cl_down_std: float
    cd_std: float
    cm_std: float
    efficiency: float
    extrapolation: list[str]
    cp_suction: list[float] | None = None
    cp_pressure: list[float] | None = None


@lru_cache(maxsize=1)
def _program() -> torch.nn.Module:
    return torch.export.load(str(ARTIFACTS / "ensemble.pt2")).module()


@lru_cache(maxsize=1)
def _metrics() -> dict:  # type: ignore[type-arg]
    path = ARTIFACTS / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else {}


@app.get("/health")
def health() -> dict[str, str]:
    _program()
    return {"status": "ok"}


@app.get("/model")
def model_info() -> dict:  # type: ignore[type-arg]
    m = _metrics()
    return {
        "dataset": m.get("dataset"),
        "passed_gates": m.get("passed"),
        "design_space": m.get("design_space"),
        "x_c": cp_stations().tolist(),
    }


@app.post("/predict")
def predict(req: PredictRequest) -> list[Prediction]:
    x = np.array([[getattr(d, p) for p in PARAMS] for d in req.designs], dtype=np.float32)
    with torch.no_grad():
        out = _program()(torch.from_numpy(x)).numpy()
    bounds = DesignSpace().bounds()
    n = len(cp_stations())
    preds = []
    for row, xi in zip(out, x, strict=True):
        mu, sd, cp = row[:3], row[3:6], row[6 : 6 + 2 * n]
        outside = [p for p, v, (lo, hi) in zip(PARAMS, xi, bounds, strict=True) if not lo <= v <= hi]
        preds.append(
            Prediction(
                cl_down=float(mu[0]),
                cd=float(mu[1]),
                cm=float(mu[2]),
                cl_down_std=float(sd[0]),
                cd_std=float(sd[1]),
                cm_std=float(sd[2]),
                efficiency=float(mu[0] / mu[1]),
                extrapolation=outside,
                cp_suction=cp[:n].tolist() if req.include_cp else None,
                cp_pressure=cp[n:].tolist() if req.include_cp else None,
            )
        )
    return preds
