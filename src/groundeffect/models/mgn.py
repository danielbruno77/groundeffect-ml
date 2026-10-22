"""Graph-network field surrogate on the OpenFOAM cell graph.

Nodes are cell centres in a window around the wing, edges are the mesh's
internal faces (both directions). Inputs per node: position, wall distance and
the design variables; outputs: u/U, v/U and Cp. The architecture is
MeshGraphNet (Pfaff et al., ICLR 2021): encode -> N message-passing blocks with
residual edge/node updates -> decode.

Backends:
* ``physicsnemo``: NVIDIA PhysicsNeMo's MeshGraphNet (needs torch_geometric and
  torch_scatter; the production path on Linux + GPU),
* ``torch``: a compact pure-PyTorch implementation of the same architecture,
  for CPU/macOS development and unit tests.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from groundeffect.doe import PARAMS, DesignSpace

OUTPUTS = ("u", "v", "cp")


# --------------------------------------------------------------------------- data
@dataclass
class Graph:
    case_id: str
    pos: torch.Tensor  # [N, 2] chord units
    node_x: torch.Tensor  # [N, F] raw node features
    edge_index: torch.Tensor  # [2, E] directed
    edge_x: torch.Tensor  # [E, 3] (dx, dy, |d|)
    y: torch.Tensor  # [N, 3] targets


def load_graph(case_dir: Path, space: DesignSpace | None = None) -> Graph:
    space = space or DesignSpace()
    z = np.load(case_dir / "fields.npz")
    design = json.loads((case_dir / "result.json").read_text())["design"]
    u = space.to_unit(np.array([[design[p] for p in PARAMS]]))[0] * 2 - 1
    pos = z["graph_pos"].astype(np.float32)
    n = len(pos)
    node_x = np.column_stack([pos, np.log10(z["graph_wall_dist"] + 1e-4), np.tile(u, (n, 1))]).astype(np.float32)
    e = z["graph_edges"].astype(np.int64)
    ei = np.hstack([e.T, e[:, ::-1].T])  # [2, 2E]: both directions
    d = pos[ei[1]] - pos[ei[0]]
    edge_x = np.column_stack([d, np.linalg.norm(d, axis=1)]).astype(np.float32)
    return Graph(
        case_id=case_dir.name,
        pos=torch.from_numpy(pos),
        node_x=torch.from_numpy(node_x),
        edge_index=torch.from_numpy(ei),
        edge_x=torch.from_numpy(edge_x),
        y=torch.from_numpy(z["graph_y"].astype(np.float32)),
    )


@dataclass
class Normaliser:
    node_mean: torch.Tensor
    node_std: torch.Tensor
    edge_mean: torch.Tensor
    edge_std: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor

    @classmethod
    def fit(cls, graphs: list[Graph]) -> Normaliser:
        nx = torch.cat([g.node_x for g in graphs])
        ex = torch.cat([g.edge_x for g in graphs])
        yy = torch.cat([g.y for g in graphs])
        eps = 1e-6
        return cls(nx.mean(0), nx.std(0) + eps, ex.mean(0), ex.std(0) + eps, yy.mean(0), yy.std(0) + eps)

    def state(self) -> dict[str, list[float]]:
        return {k: v.tolist() for k, v in asdict(self).items()}

    @classmethod
    def from_state(cls, s: dict[str, list[float]]) -> Normaliser:
        return cls(**{k: torch.tensor(v) for k, v in s.items()})


# ------------------------------------------------------------------------- models
def _mlp(n_in: int, n_out: int, hidden: int, layers: int = 2, norm: bool = True) -> nn.Sequential:
    mods: list[nn.Module] = []
    d = n_in
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    mods.append(nn.Linear(d, n_out))
    if norm:
        mods.append(nn.LayerNorm(n_out))
    return nn.Sequential(*mods)


class TorchMeshGraphNet(nn.Module):
    """Encode-process-decode MeshGraphNet with sum aggregation (pure PyTorch)."""

    def __init__(self, node_in: int, edge_in: int, out: int, hidden: int = 64, steps: int = 8) -> None:
        super().__init__()
        self.node_enc = _mlp(node_in, hidden, hidden)
        self.edge_enc = _mlp(edge_in, hidden, hidden)
        self.edge_mlps = nn.ModuleList([_mlp(3 * hidden, hidden, hidden) for _ in range(steps)])
        self.node_mlps = nn.ModuleList([_mlp(2 * hidden, hidden, hidden) for _ in range(steps)])
        self.dec = _mlp(hidden, out, hidden, norm=False)

    def forward(self, node_x: torch.Tensor, edge_x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        h, e = self.node_enc(node_x), self.edge_enc(edge_x)
        for edge_mlp, node_mlp in zip(self.edge_mlps, self.node_mlps, strict=True):
            e = e + edge_mlp(torch.cat([e, h[src], h[dst]], dim=1))
            agg = torch.zeros_like(h).index_add_(0, dst, e)
            h = h + node_mlp(torch.cat([h, agg], dim=1))
        return self.dec(h)


def physicsnemo_available() -> bool:
    try:
        os.environ.setdefault("PHYSICSNEMO_FORCE_TE", "False")
        import torch_scatter  # noqa: F401
        from physicsnemo.models.meshgraphnet import MeshGraphNet  # noqa: F401
        from torch_geometric.data import Data  # noqa: F401
    except Exception:
        return False
    return True


class FieldModel(nn.Module):
    """Uniform call signature over both backends."""

    def __init__(self, backend: str, node_in: int, edge_in: int, out: int, hidden: int, steps: int) -> None:
        super().__init__()
        self.backend = backend
        if backend == "physicsnemo":
            from physicsnemo.models.meshgraphnet import MeshGraphNet

            self.net: nn.Module = MeshGraphNet(
                input_dim_nodes=node_in,
                input_dim_edges=edge_in,
                output_dim=out,
                processor_size=steps,
                hidden_dim_processor=hidden,
                hidden_dim_node_encoder=hidden,
                hidden_dim_edge_encoder=hidden,
                hidden_dim_node_decoder=hidden,
            )
        else:
            self.net = TorchMeshGraphNet(node_in, edge_in, out, hidden, steps)

    def forward(self, node_x: torch.Tensor, edge_x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.backend == "physicsnemo":
            from torch_geometric.data import Data

            return self.net(node_x, edge_x, Data(edge_index=edge_index, num_nodes=node_x.shape[0]))
        return self.net(node_x, edge_x, edge_index)


# ----------------------------------------------------------------------- training
@dataclass
class MGNConfig:
    backend: str = "auto"
    hidden: int = 64
    steps: int = 8
    epochs: int = 30
    lr: float = 1e-3
    seed: int = 0
    device: str = "cpu"
    history: list[float] = field(default_factory=list)


def train_mgn(graphs: list[Graph], cfg: MGNConfig, log_every: int = 1) -> tuple[FieldModel, Normaliser]:
    torch.manual_seed(cfg.seed)
    backend = cfg.backend if cfg.backend != "auto" else ("physicsnemo" if physicsnemo_available() else "torch")
    cfg.backend = backend
    norm = Normaliser.fit(graphs)
    g0 = graphs[0]
    model = FieldModel(backend, g0.node_x.shape[1], g0.edge_x.shape[1], len(OUTPUTS), cfg.hidden, cfg.steps).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs * len(graphs))
    rng = np.random.default_rng(cfg.seed)
    for epoch in range(cfg.epochs):
        t0, total = time.time(), 0.0
        for i in rng.permutation(len(graphs)):
            g = graphs[i]
            nx = ((g.node_x - norm.node_mean) / norm.node_std).to(cfg.device)
            ex = ((g.edge_x - norm.edge_mean) / norm.edge_std).to(cfg.device)
            yy = ((g.y - norm.y_mean) / norm.y_std).to(cfg.device)
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(nx, ex, g.edge_index.to(cfg.device)), yy)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += float(loss)
        cfg.history.append(total / len(graphs))
        if log_every and epoch % log_every == 0:
            print(f"epoch {epoch:3d}  loss {cfg.history[-1]:.4f}  {time.time() - t0:.1f}s  [{backend}]", flush=True)
    return model.eval(), norm


@torch.no_grad()
def predict_graph(model: FieldModel, norm: Normaliser, g: Graph, device: str = "cpu") -> torch.Tensor:
    nx = ((g.node_x - norm.node_mean) / norm.node_std).to(device)
    ex = ((g.edge_x - norm.edge_mean) / norm.edge_std).to(device)
    out = model(nx, ex, g.edge_index.to(device)).cpu()
    return out * norm.y_std + norm.y_mean


def relative_l2(pred: torch.Tensor, true: torch.Tensor) -> list[float]:
    """Per-output relative L2 error, the standard field-surrogate metric."""
    return [float(torch.linalg.norm(pred[:, k] - true[:, k]) / torch.linalg.norm(true[:, k])) for k in range(true.shape[1])]


def save_mgn(model: FieldModel, norm: Normaliser, cfg: MGNConfig, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "mgn.pt")
    (path / "mgn.json").write_text(json.dumps({"config": asdict(cfg), "normaliser": norm.state()}, indent=1))


def load_mgn(path: Path) -> tuple[FieldModel, Normaliser, MGNConfig]:
    meta = json.loads((path / "mgn.json").read_text())
    cfg = MGNConfig(**meta["config"])
    norm = Normaliser.from_state(meta["normaliser"])
    node_in, edge_in = len(meta["normaliser"]["node_mean"]), len(meta["normaliser"]["edge_mean"])
    model = FieldModel(cfg.backend, node_in, edge_in, len(OUTPUTS), cfg.hidden, cfg.steps)
    model.load_state_dict(torch.load(path / "mgn.pt", weights_only=True))
    return model.eval(), norm, cfg
