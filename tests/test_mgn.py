import numpy as np
import pytest
import torch

from groundeffect.models.mgn import FieldModel, Graph, MGNConfig, physicsnemo_available, predict_graph, relative_l2, train_mgn


def tiny_graph(n=60, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, 1, (n, 2)).astype(np.float32)
    edges = np.array([(i, j) for i in range(n) for j in range(i + 1, n) if np.linalg.norm(pos[i] - pos[j]) < 0.25])
    ei = np.hstack([edges.T, edges[:, ::-1].T])
    d = pos[ei[1]] - pos[ei[0]]
    node_x = np.column_stack([pos, np.zeros((n, 5))]).astype(np.float32)
    y = np.column_stack([np.sin(3 * pos[:, 0]), pos[:, 1], pos[:, 0] * pos[:, 1]]).astype(np.float32)
    return Graph(
        "toy",
        torch.from_numpy(pos),
        torch.from_numpy(node_x),
        torch.from_numpy(ei),
        torch.from_numpy(np.column_stack([d, np.linalg.norm(d, axis=1)]).astype(np.float32)),
        torch.from_numpy(y),
    )


def test_torch_backend_trains():
    g = tiny_graph()
    model, norm = train_mgn([g], MGNConfig(backend="torch", hidden=16, steps=2, epochs=150, lr=3e-3), log_every=0)
    err = relative_l2(predict_graph(model, norm, g), g.y)
    assert max(err) < 0.3


@pytest.mark.skipif(not physicsnemo_available(), reason="PhysicsNeMo graph stack not installed")
def test_fallback_matches_physicsnemo_architecture():
    a = FieldModel("physicsnemo", 7, 3, 3, 32, 4)
    b = FieldModel("torch", 7, 3, 3, 32, 4)
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())
    g = tiny_graph()
    assert a(g.node_x, g.edge_x, g.edge_index).shape == (len(g.pos), 3)
