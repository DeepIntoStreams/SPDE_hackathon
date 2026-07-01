from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import scipy.io
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset


class LpLoss:
    def __init__(self, p: int = 2, size_average: bool = True):
        self.p = p
        self.size_average = size_average

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n = pred.shape[0]
        diff = torch.norm(pred.reshape(n, -1) - target.reshape(n, -1), self.p, dim=1)
        denom = torch.norm(target.reshape(n, -1), self.p, dim=1).clamp_min(1e-12)
        rel = diff / denom
        return rel.mean() if self.size_average else rel.sum()


def read_mat_any(path: str | Path) -> dict:
    path = Path(path)
    try:
        return {k: v for k, v in scipy.io.loadmat(path).items() if not k.startswith("__")}
    except NotImplementedError:
        out = {}
        with h5py.File(path, "r") as f:
            for key in f.keys():
                arr = f[key][()]
                out[key] = np.transpose(arr, axes=range(arr.ndim - 1, -1, -1)) if arr.ndim > 1 else arr
        return out


def _squeeze_grid(arr, fallback_len: int | None = None):
    if arr is None:
        if fallback_len is None:
            raise ValueError("Missing grid and no fallback length was provided.")
        return np.linspace(0.0, 1.0, fallback_len, dtype=np.float32)
    arr = np.asarray(arr).squeeze()
    if arr.ndim == 2:
        arr = arr[:, 0]
    return arr.astype(np.float32)


def _select_time(arr: np.ndarray, max_t: int | None, sub_t: int) -> np.ndarray:
    stop = arr.shape[1] if max_t is None else min(max_t, arr.shape[1])
    return arr[:, :stop:sub_t]


def load_spde_1d(path: str | Path, *, max_t: int | None = None, sub_t: int = 1, sub_x: int = 1) -> dict:
    data = read_mat_any(path)
    if "W" not in data or "sol" not in data:
        raise KeyError(f"{path} must contain unified keys 'W' and 'sol'.")
    W = np.asarray(data["W"], dtype=np.float32)
    U = np.asarray(data["sol"], dtype=np.float32)
    T = _squeeze_grid(data.get("T"), W.shape[-1])

    if W.ndim != 3 or U.ndim != 3:
        raise ValueError(f"Expected 1D W/sol with 3 dims, got {W.shape} and {U.shape}.")
    if W.shape[1] == len(T):
        W_tx, U_tx = W, U
    elif W.shape[2] == len(T):
        W_tx, U_tx = np.transpose(W, (0, 2, 1)), np.transpose(U, (0, 2, 1))
    else:
        raise ValueError(f"Cannot infer time axis from W shape={W.shape}, len(T)={len(T)}")

    W_tx = _select_time(W_tx[:, :, ::sub_x], max_t, sub_t)
    U_tx = _select_time(U_tx[:, :, ::sub_x], max_t, sub_t)
    T = T[: W_tx.shape[1] * sub_t : sub_t]
    X = _squeeze_grid(data.get("X"), W_tx.shape[2])[::sub_x]
    return {"W": W_tx, "sol": U_tx, "X": X, "T": T}


def load_spde_2d(path: str | Path, *, max_t: int | None = None, sub_t: int = 1, sub_x: int = 1) -> dict:
    data = read_mat_any(path)
    noise_key = "W" if "W" in data else "forcing"
    if noise_key not in data or "sol" not in data:
        raise KeyError(f"{path} must contain 'W' (or 'forcing') and 'sol'.")
    W = np.asarray(data[noise_key], dtype=np.float32)
    U = np.asarray(data["sol"], dtype=np.float32)
    T = _squeeze_grid(data.get("T", data.get("t")), W.shape[-1])

    if W.ndim != 4 or U.ndim != 4:
        raise ValueError(f"Expected 2D W/sol with 4 dims, got {W.shape} and {U.shape}.")
    if W.shape[1] == len(T):
        W_txy, U_txy = W, U
    elif W.shape[-1] == len(T):
        W_txy, U_txy = np.transpose(W, (0, 3, 1, 2)), np.transpose(U, (0, 3, 1, 2))
    else:
        raise ValueError(f"Cannot infer time axis from W shape={W.shape}, len(T)={len(T)}")

    W_txy = _select_time(W_txy[:, :, ::sub_x, ::sub_x], max_t, sub_t)
    U_txy = _select_time(U_txy[:, :, ::sub_x, ::sub_x], max_t, sub_t)
    T = T[: W_txy.shape[1] * sub_t : sub_t]
    X = _squeeze_grid(data.get("X"), W_txy.shape[2])[::sub_x]
    Y = _squeeze_grid(data.get("Y"), W_txy.shape[3])[::sub_x]
    return {"W": W_txy, "sol": U_txy, "X": X, "Y": Y, "T": T}


def make_graph_1d(free_num: int = 3, include_u0_products: bool = True) -> OrderedDict:
    graph = OrderedDict()
    graph["xi"] = {}
    graph["I_c[u_0]"] = {}
    graph["I[xi]"] = {"xi": 1}
    for power in range(2, free_num + 1):
        graph[f"I[(I[xi])^{power}]"] = {"I[xi]": power}
    if include_u0_products:
        graph["I[(I_c[u_0])(I[xi])]"] = {"I_c[u_0]": 1, "I[xi]": 1}
        if free_num >= 2:
            graph["I[(I_c[u_0])(I[xi])^2]"] = {"I_c[u_0]": 1, "I[xi]": 2}
        graph["I[(I_c[u_0])^2]"] = {"I_c[u_0]": 2}
    return graph


def make_graph_2d(free_num: int = 2, include_u0_products: bool = True) -> OrderedDict:
    graph = OrderedDict()
    graph["xi"] = {}
    graph["I_c[u_0]"] = {}
    graph["I[xi]"] = {"xi": 1}
    for power in range(2, free_num + 1):
        graph[f"I[(I[xi])^{power}]"] = {"I[xi]": power}
    if include_u0_products:
        graph["I[(I_c[u_0])(I[xi])]"] = {"I_c[u_0]": 1, "I[xi]": 1}
        graph["I[(I_c[u_0])^2]"] = {"I_c[u_0]": 2}
    return graph


def make_splits(n_total: int, ntrain: int, nval: int, ntest: int):
    total = ntrain + nval + ntest
    if total > n_total:
        raise ValueError(f"Requested {total} samples but dataset only has {n_total}.")
    return range(0, ntrain), range(ntrain, ntrain + nval), range(ntrain + nval, total)


def make_loaders(W, U, ntrain: int, nval: int, ntest: int, batch_size: int, *, num_workers: int = 0):
    tensor_w = torch.from_numpy(W).float()
    tensor_u = torch.from_numpy(U).float()
    u0 = tensor_u[:, 0]
    dataset = TensorDataset(tensor_w, u0, tensor_u)
    train_idx, val_idx, test_idx = make_splits(len(dataset), ntrain, nval, ntest)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
