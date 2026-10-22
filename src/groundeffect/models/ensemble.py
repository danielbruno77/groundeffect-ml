"""PyTorch deep-ensemble surrogate: design -> (CL_down, CD, Cm, surface Cp).

Surface Cp is compressed with POD so the network predicts a handful of modal
coefficients instead of 160 correlated values; the decoded distribution is
smooth by construction. Uncertainty comes from ensemble disagreement, rescaled
on a held-out calibration split so that predicted sigma means what it says.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from groundeffect.doe import DesignSpace
from groundeffect.models.data import POD, SCALARS, Standardizer


class MLP(nn.Module):
    def __init__(self, n_in: int, n_out: int, width: int = 128, depth: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = n_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.SiLU()]
            d = width
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TrainConfig:
    members: int = 5
    width: int = 128
    depth: int = 3
    epochs: int = 3000
    lr: float = 3e-3
    weight_decay: float = 1e-5
    subsample: float = 0.9
    calibration_fraction: float = 0.15
    seed: int = 0


@dataclass
class EnsembleSurrogate:
    config: TrainConfig = field(default_factory=TrainConfig)
    space: DesignSpace = field(default_factory=DesignSpace)
    members: list[MLP] = field(default_factory=list)
    y_scaler: Standardizer | None = None
    pod: POD | None = None
    c_scaler: Standardizer | None = None
    sigma_scale: np.ndarray | None = None  # per-output calibration factor
    history: list[list[float]] = field(default_factory=list)

    # ---- features -------------------------------------------------------
    def _inputs(self, x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(self.space.to_unit(x) * 2.0 - 1.0, dtype=torch.float32)

    def _targets(self, y: np.ndarray, cp: np.ndarray) -> np.ndarray:
        assert self.y_scaler and self.pod and self.c_scaler
        return np.hstack([self.y_scaler.transform(y), self.c_scaler.transform(self.pod.encode(cp))])

    # ---- training -------------------------------------------------------
    def fit(self, x: np.ndarray, y: np.ndarray, cp: np.ndarray) -> EnsembleSurrogate:
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        torch.manual_seed(cfg.seed)
        n = len(x)
        perm = rng.permutation(n)
        n_cal = max(4, int(cfg.calibration_fraction * n))
        cal, fit_idx = perm[:n_cal], perm[n_cal:]

        self.y_scaler = Standardizer.fit(y[fit_idx])
        self.pod = POD.fit(cp[fit_idx])
        self.c_scaler = Standardizer.fit(self.pod.encode(cp[fit_idx]))
        xt, zt = self._inputs(x), torch.as_tensor(self._targets(y, cp), dtype=torch.float32)

        self.members, self.history = [], []
        for m in range(cfg.members):
            sub = rng.choice(fit_idx, int(cfg.subsample * len(fit_idx)), replace=False)
            net = MLP(xt.shape[1], zt.shape[1], cfg.width, cfg.depth)
            opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)
            losses = []
            for epoch in range(cfg.epochs):
                opt.zero_grad()
                loss = nn.functional.mse_loss(net(xt[sub]), zt[sub])
                loss.backward()
                opt.step()
                sched.step()
                if epoch % 50 == 0 or epoch == cfg.epochs - 1:
                    losses.append(loss.item())
            self.members.append(net.eval())
            self.history.append(losses)
            del m

        # Variance calibration on the held-out split: scale sigma so the
        # standardised residuals have unit RMS.
        mu, sd = self._raw_predict(xt[cal])
        z = (mu - zt[cal].numpy()) / np.maximum(sd, 1e-9)
        self.sigma_scale = np.sqrt(np.mean(z**2, axis=0)).clip(0.5, 20.0)
        return self

    # ---- inference ------------------------------------------------------
    @torch.no_grad()
    def _raw_predict(self, xt: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        preds = torch.stack([m(xt) for m in self.members]).numpy()
        return preds.mean(0), preds.std(0)

    def predict(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """Physical-unit predictions with 1-sigma uncertainty."""
        assert self.y_scaler and self.pod and self.c_scaler and self.sigma_scale is not None
        with torch.no_grad():
            preds = torch.stack([m(self._inputs(x)) for m in self.members]).numpy()
        ns = len(SCALARS)
        scal = preds[..., :ns] * self.y_scaler.std + self.y_scaler.mean
        coef = preds[..., ns:] * self.c_scaler.std + self.c_scaler.mean
        cp = np.stack([self.pod.decode(c) for c in coef])
        s_y = self.sigma_scale[:ns]
        s_c = float(np.sqrt(np.mean(self.sigma_scale[ns:] ** 2)))
        return {
            "scalars": scal.mean(0),
            "scalars_std": scal.std(0) * s_y,
            "cp": cp.mean(0),
            "cp_std": cp.std(0) * s_c,
        }

    # ---- persistence ----------------------------------------------------
    def save(self, path: Path) -> None:
        assert self.y_scaler and self.pod and self.c_scaler and self.sigma_scale is not None
        path.mkdir(parents=True, exist_ok=True)
        torch.save([m.state_dict() for m in self.members], path / "members.pt")
        np.savez(
            path / "normalisation.npz",
            y_mean=self.y_scaler.mean,
            y_std=self.y_scaler.std,
            pod_mean=self.pod.mean,
            pod_modes=self.pod.modes,
            pod_energy=self.pod.energy,
            c_mean=self.c_scaler.mean,
            c_std=self.c_scaler.std,
            sigma_scale=self.sigma_scale,
        )
        (path / "config.json").write_text(json.dumps(asdict(self.config), indent=2))

    @classmethod
    def load(cls, path: Path) -> EnsembleSurrogate:
        cfg = TrainConfig(**json.loads((path / "config.json").read_text()))
        z = np.load(path / "normalisation.npz")
        obj = cls(config=cfg)
        obj.y_scaler = Standardizer(z["y_mean"], z["y_std"])
        obj.pod = POD(z["pod_mean"], z["pod_modes"], z["pod_energy"])
        obj.c_scaler = Standardizer(z["c_mean"], z["c_std"])
        obj.sigma_scale = z["sigma_scale"]
        n_out = len(SCALARS) + obj.pod.k
        for sd in torch.load(path / "members.pt", weights_only=True):
            net = MLP(4, n_out, cfg.width, cfg.depth)
            net.load_state_dict(sd)
            obj.members.append(net.eval())
        return obj


class ExportableEnsemble(nn.Module):
    """Self-contained module: raw design variables in, physical outputs out.

    Normalisation, POD decoding and calibration are baked in as buffers so the
    exported program has no Python-side preprocessing to drift out of sync.
    Output: [mean scalars (3), std scalars (3), mean Cp (160), std Cp (160)].
    """

    lo: torch.Tensor
    hi: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor
    c_mean: torch.Tensor
    c_std: torch.Tensor
    pod_mean: torch.Tensor
    pod_modes: torch.Tensor
    s_y: torch.Tensor

    def __init__(self, surrogate: EnsembleSurrogate) -> None:
        super().__init__()
        assert surrogate.y_scaler and surrogate.pod and surrogate.c_scaler and surrogate.sigma_scale is not None
        self.members = nn.ModuleList([copy.deepcopy(m) for m in surrogate.members])
        b = surrogate.space.bounds()
        f32 = torch.float32
        self.register_buffer("lo", torch.tensor(b[:, 0], dtype=f32))
        self.register_buffer("hi", torch.tensor(b[:, 1], dtype=f32))
        self.log_h = surrogate.space.log_ride_height
        self.register_buffer("y_mean", torch.tensor(surrogate.y_scaler.mean, dtype=f32))
        self.register_buffer("y_std", torch.tensor(surrogate.y_scaler.std, dtype=f32))
        self.register_buffer("c_mean", torch.tensor(surrogate.c_scaler.mean, dtype=f32))
        self.register_buffer("c_std", torch.tensor(surrogate.c_scaler.std, dtype=f32))
        self.register_buffer("pod_mean", torch.tensor(surrogate.pod.mean, dtype=f32))
        self.register_buffer("pod_modes", torch.tensor(surrogate.pod.modes, dtype=f32))
        ns = len(SCALARS)
        self.ns = ns
        self.register_buffer("s_y", torch.tensor(surrogate.sigma_scale[:ns], dtype=f32))
        self.s_c = float(np.sqrt(np.mean(surrogate.sigma_scale[ns:] ** 2)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = (x - self.lo) / (self.hi - self.lo)
        if self.log_h:
            h = (torch.log(x[:, 0]) - torch.log(self.lo[0])) / (torch.log(self.hi[0]) - torch.log(self.lo[0]))
            u = torch.cat([h[:, None], u[:, 1:]], dim=1)
        z = u * 2.0 - 1.0
        preds = torch.stack([m(z) for m in self.members])
        scal = preds[..., : self.ns] * self.y_std + self.y_mean
        coef = preds[..., self.ns :] * self.c_std + self.c_mean
        cp = coef @ self.pod_modes + self.pod_mean
        return torch.cat([scal.mean(0), scal.std(0, correction=0) * self.s_y, cp.mean(0), cp.std(0, correction=0) * self.s_c], dim=1)


def export_program(surrogate: EnsembleSurrogate, path: Path) -> Path:
    """Export with ``torch.export`` (portable .pt2 program, dynamic batch)."""
    module = ExportableEnsemble(surrogate).eval()
    example = torch.tensor([[0.15, 4.0, 0.04, 0.12], [0.3, 2.0, 0.03, 0.1]], dtype=torch.float32)
    batch = torch.export.Dim("batch", min=1, max=4096)
    program = torch.export.export(module, (example,), dynamic_shapes=({0: batch},))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(program, path)
    return path
