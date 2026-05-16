from __future__ import annotations

from torch.utils.data import DataLoader, TensorDataset
import torch

from model.NORS.Graph import Graph
from model.NORS.RSlayer import ParabolicIntegrate
from model.NORS.Rule import Rule
from model.NORS.SPDEs import SPDE
from model.NORS.utils import load_spde_1d


def parabolic_graph(data, height=2, kernel_deg=2, noise_deg=-1.5, free_num=3, deg=7.5):
    """Build the DLR/NORS non-singular 1D parabolic MFV graph."""
    rule = Rule(kernel_deg=kernel_deg, noise_deg=noise_deg, free_num=free_num)
    integration = SPDE(BC="P", T=data["T"], X=data["X"]).Integrate_Parabolic_trees
    graph_builder = Graph(integration=integration, rule=rule, height=height, deg=deg)
    key = "I_c[u_0]"
    extra_deg = {key: kernel_deg}
    graph = graph_builder.create_model_graph(
        data["W"][0:1],
        extra_planted={key: data["W"][0:1]},
        extra_deg=extra_deg,
    )
    return graph


def cacheXiFeature(graph, T, X, W, device, batch_size=100, eps=1.0, boundary="P", diff=True, noise_scale=1.0):
    """DLR-compatible cache of Xi-only 1D model feature vectors.

    Returns a tensor with shape [N, T, X, C], where C is len(graph). Features
    involving u0 are present as zero channels, matching the graph indices used
    later when full features are built with a latent initial-condition path.
    """
    W = torch.as_tensor(W).float() * noise_scale
    layer = ParabolicIntegrate(graph, T=T, X=X, eps=eps, BC=boundary).to(device)
    loader = DataLoader(TensorDataset(W), batch_size=batch_size, shuffle=False)
    chunks = []
    with torch.no_grad():
        for (batch_w,) in loader:
            chunks.append(layer(W=batch_w.to(device), diff=diff).cpu())
    return torch.cat(chunks, dim=0)


def build_mfv_1d(
    graph,
    T,
    X,
    W,
    U0=None,
    device=None,
    batch_size=100,
    eps=1.0,
    boundary="P",
    diff=True,
    noise_scale=1.0,
):
    """Build full 1D non-singular MFV, optionally including u0-dependent features."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
    W = torch.as_tensor(W).float() * noise_scale
    U0 = None if U0 is None else torch.as_tensor(U0).float()
    layer = ParabolicIntegrate(graph, T=T, X=X, eps=eps, BC=boundary).to(device)
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


def build_mfv_1d_from_mat(
    data_path,
    *,
    max_t=None,
    sub_t=1,
    sub_x=1,
    height=2,
    free_num=3,
    batch_size=100,
    device=None,
    eps=1.0,
    boundary="P",
    include_u0=True,
):
    """Convenience entrypoint: unified MAT file -> 1D MFV tensor and target."""
    data = load_spde_1d(data_path, max_t=max_t, sub_t=sub_t, sub_x=sub_x)
    graph = parabolic_graph(data, height=height, free_num=free_num)
    u0 = data["sol"][:, 0] if include_u0 else None
    mfv = build_mfv_1d(
        graph,
        data["T"],
        data["X"],
        data["W"],
        U0=u0,
        device=device,
        batch_size=batch_size,
        eps=eps,
        boundary=boundary,
        noise_scale=0.1,
    )
    return {"x": mfv.numpy(), "u": data["sol"], "graph": graph, **data}
