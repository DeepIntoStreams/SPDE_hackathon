from __future__ import annotations

from torch.utils.data import DataLoader, TensorDataset
import torch

from model.NORS.Graph import Graph
from model.NORS.RSlayer_2d import ParabolicIntegrate_2d
from model.NORS.Rule import Rule
from model.NORS.SPDEs import SPDE
from model.NORS.utils import load_spde_2d


def parabolic_graph_2d(data, height=2, kernel_deg=2, noise_deg=-2.0, free_num=2, deg=7.5):
    """Build the DLR/NORS non-singular 2D parabolic MFV graph."""
    rule = Rule(kernel_deg=kernel_deg, noise_deg=noise_deg, free_num=free_num)
    mesh_x, mesh_y = torch_mesh_to_numpy_mesh(data["X"], data["Y"])
    integration = SPDE(BC="P", T=data["T"], X=mesh_x, Y=mesh_y).Integrate_Parabolic_trees_2d
    graph_builder = Graph(integration=integration, rule=rule, height=height, deg=deg)
    key = "I_c[u_0]"
    extra_deg = {key: kernel_deg}
    graph = graph_builder.create_model_graph_2d(
        data["W"][0:1],
        mesh_x,
        extra_planted={key: data["W"][0:1]},
        extra_deg=extra_deg,
    )
    return graph


def torch_mesh_to_numpy_mesh(X, Y):
    import numpy as np

    x = np.asarray(X)
    y = np.asarray(Y)
    if x.ndim == 1 and y.ndim == 1:
        return np.meshgrid(x, y, indexing="ij")
    return x, y


def cacheXiFeature_2d(graph, T, X, Y, W, eps, device, batch_size=100, diff=False):
    """DLR-compatible cache of Xi-only 2D model feature vectors."""
    layer = ParabolicIntegrate_2d(graph, T=T, X=X, Y=Y, eps=eps).to(device)
    loader = DataLoader(TensorDataset(W), batch_size=batch_size, shuffle=False)
    chunks = []
    with torch.no_grad():
        for (batch_w,) in loader:
            chunks.append(layer(W=batch_w.to(device), diff=diff).cpu())
    return torch.cat(chunks, dim=0)


def build_mfv_2d(graph, T, X, Y, W, U0=None, device=None, batch_size=100, eps=1.0, diff=False):
    """Build full 2D non-singular MFV, optionally including u0-dependent features."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
    W = torch.as_tensor(W).float()
    U0 = None if U0 is None else torch.as_tensor(U0).float()
    layer = ParabolicIntegrate_2d(graph, T=T, X=X, Y=Y, eps=eps).to(device)
    if U0 is None:
        dataset = TensorDataset(W)
    else:
        dataset = TensorDataset(W, U0)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    chunks = []
    with torch.no_grad():
        for batch in loader:
            if U0 is None:
                (batch_w,) = batch
                chunks.append(layer(W=batch_w.to(device), diff=diff).cpu())
            else:
                batch_w, batch_u0 = batch
                latent = layer.I_c(batch_u0.to(device))
                chunks.append(layer(W=batch_w.to(device), U0_path=latent, diff=diff).cpu())
    return torch.cat(chunks, dim=0)


def build_mfv_2d_from_mat(
    data_path,
    *,
    max_t=None,
    sub_t=1,
    sub_x=1,
    height=2,
    free_num=2,
    batch_size=100,
    device=None,
    eps=1.0,
    include_u0=True,
):
    """Convenience entrypoint: unified MAT file -> 2D MFV tensor and target."""
    data = load_spde_2d(data_path, max_t=max_t, sub_t=sub_t, sub_x=sub_x)
    graph = parabolic_graph_2d(data, height=height, free_num=free_num)
    u0 = data["sol"][:, 0] if include_u0 else None
    mfv = build_mfv_2d(
        graph,
        data["T"],
        data["X"],
        data["Y"],
        data["W"],
        U0=u0,
        device=device,
        batch_size=batch_size,
        eps=eps,
    )
    return {"x": mfv.numpy(), "u": data["sol"], "graph": graph, **data}
