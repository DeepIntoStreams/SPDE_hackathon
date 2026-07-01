from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn


def parabolic_matrix_1d(n_points: int, dt: float, dx: float, eps: float, bc: str, *, device=None, dtype=None):
    kwargs = {"device": device, "dtype": dtype}
    n = n_points - 1
    mat = torch.diag(-2 * torch.ones(n + 1, **kwargs))
    mat += torch.diag(torch.ones(n, **kwargs), diagonal=1)
    mat += torch.diag(torch.ones(n, **kwargs), diagonal=-1)
    if bc == "D":
        mat[0, 0], mat[0, 1], mat[1, 0] = 0, 0, 0
        mat[-1, -1], mat[-1, -2], mat[-2, -1] = 0, 0, 0
    elif bc == "N":
        mat[0, 1], mat[-1, -2] = 2, 2
    elif bc == "P":
        mat[-1, 1], mat[0, -2] = 1, 1
    else:
        raise ValueError(f"Unknown boundary condition: {bc}")
    eye = torch.eye(n + 1, **kwargs)
    return torch.linalg.inv(eye - eps * dt * mat / (dx**2))


class ParabolicIntegrate1D(nn.Module):
    """DLR-style parabolic feature layer for 1D non-singular SPDEs."""

    def __init__(self, graph: OrderedDict, t_grid, x_grid, *, bc: str = "P", eps: float = 1.0):
        super().__init__()
        self.graph = OrderedDict(graph)
        self.keys = list(self.graph)
        self.name_to_index = {name: idx for idx, name in enumerate(self.keys)}
        self.bc = bc
        self.eps = eps
        self.t_grid = torch.as_tensor(t_grid, dtype=torch.float32)
        self.x_grid = torch.as_tensor(x_grid, dtype=torch.float32)
        self.nt = len(self.t_grid)
        self.nx = len(self.x_grid)
        self.dt = float(self.t_grid[1] - self.t_grid[0])
        self.dx = float(self.x_grid[1] - self.x_grid[0])

        mat = parabolic_matrix_1d(self.nx, self.dt, self.dx, eps, bc, dtype=torch.float32).T
        mat_ic = parabolic_matrix_1d(self.nx, self.dt, self.dx, eps, "D", dtype=torch.float32).T
        self.register_buffer("mat", mat)
        self.register_buffer("mat_ic", mat_ic)

    def I_c(self, u0: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(u0.shape[0], self.nt, self.nx, device=u0.device, dtype=u0.dtype)
        out[:, 0] = u0
        mat = self.mat_ic.to(device=u0.device, dtype=u0.dtype)
        for i in range(1, self.nt):
            out[:, i] = out[:, i - 1] @ mat
        return out

    def heat_integrate(self, source: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(source)
        mat = self.mat.to(device=source.device, dtype=source.dtype)
        for i in range(1, self.nt):
            out[:, i] = out[:, i - 1] @ mat + source[:, i] * self.dt
        return out

    def forward(
        self,
        W: torch.Tensor,
        U0_path: torch.Tensor | None = None,
        XiFeature: torch.Tensor | None = None,
        *,
        diff: bool = False,
    ) -> torch.Tensor:
        if diff:
            xi = torch.zeros_like(W)
            xi[:, 1:] = torch.diff(W, dim=1) / self.dt
        else:
            xi = W

        features: list[torch.Tensor] = []
        for idx, (name, deps) in enumerate(self.graph.items()):
            if XiFeature is not None and "u_0" not in name:
                features.append(XiFeature[..., idx])
            elif name == "xi":
                features.append(xi)
            elif name == "I_c[u_0]":
                if U0_path is None:
                    features.append(torch.zeros_like(xi))
                else:
                    features.append(U0_path)
            else:
                source = torch.ones_like(xi)
                for dep_name, power in deps.items():
                    source = source * features[self.name_to_index[dep_name]].pow(power)
                features.append(self.heat_integrate(source))
        return torch.stack(features, dim=-1)


class ParabolicIntegrate2D(nn.Module):
    """DLR-style parabolic feature layer for 2D non-singular SPDEs."""

    def __init__(self, graph: OrderedDict, t_grid, x_grid, y_grid, *, eps: float = 1.0):
        super().__init__()
        self.graph = OrderedDict(graph)
        self.keys = list(self.graph)
        self.name_to_index = {name: idx for idx, name in enumerate(self.keys)}
        self.eps = eps
        self.t_grid = torch.as_tensor(t_grid, dtype=torch.float32)
        self.x_grid = torch.as_tensor(x_grid, dtype=torch.float32)
        self.y_grid = torch.as_tensor(y_grid, dtype=torch.float32)
        self.nt = len(self.t_grid)
        self.nx = len(self.x_grid)
        self.ny = len(self.y_grid)
        self.dt = float(self.t_grid[1] - self.t_grid[0])
        self.dx = float(self.x_grid[1] - self.x_grid[0])
        self.dy = float(self.y_grid[1] - self.y_grid[0])

        lap = torch.tensor([[[[0.25, 0.5, 0.25], [0.5, -3.0, 0.5], [0.25, 0.5, 0.25]]]], dtype=torch.float32)
        filt = torch.tensor([[[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]], dtype=torch.float32)
        filt = filt + eps * lap * self.dt / (self.dx**2)
        self.register_buffer("filter_i", filt)

    def laplace_i(self, arr: torch.Tensor) -> torch.Tensor:
        filt = self.filter_i.to(device=arr.device, dtype=arr.dtype)
        return F.conv2d(F.pad(arr.unsqueeze(1), (1, 1, 1, 1), mode="circular"), filt).squeeze(1)

    def I_c(self, u0: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(u0.shape[0], self.nt, self.nx, self.ny, device=u0.device, dtype=u0.dtype)
        out[:, 0] = u0
        for i in range(1, self.nt):
            out[:, i] = self.laplace_i(out[:, i - 1])
        return out

    def heat_integrate(self, source: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(source)
        for i in range(1, self.nt):
            out[:, i] = self.laplace_i(out[:, i - 1]) + source[:, i] * self.dt
        return out

    def forward(
        self,
        W: torch.Tensor,
        U0_path: torch.Tensor | None = None,
        XiFeature: torch.Tensor | None = None,
        *,
        diff: bool = False,
    ) -> torch.Tensor:
        if diff:
            xi = torch.zeros_like(W)
            xi[:, 1:] = torch.diff(W, dim=1) / self.dt
        else:
            xi = W

        features: list[torch.Tensor] = []
        for idx, (name, deps) in enumerate(self.graph.items()):
            if XiFeature is not None and "u_0" not in name:
                features.append(XiFeature[..., idx])
            elif name == "xi":
                features.append(xi)
            elif name == "I_c[u_0]":
                if U0_path is None:
                    features.append(torch.zeros_like(xi))
                else:
                    features.append(U0_path)
            else:
                source = torch.ones_like(xi)
                for dep_name, power in deps.items():
                    source = source * features[self.name_to_index[dep_name]].pow(power)
                features.append(self.heat_integrate(source))
        return torch.stack(features, dim=-1)


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def compl_mul2d(self, inputs, weights):
        return torch.einsum("bixy,ioxy->boxy", inputs, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batchsize,
            self.weights1.shape[1],
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2],
            self.weights1,
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2],
            self.weights2,
        )
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class FNO2DLayer(nn.Module):
    def __init__(self, modes1: int, modes2: int, width: int, *, last: bool = False):
        super().__init__()
        self.last = last
        self.conv = SpectralConv2d(width, width, modes1, modes2)
        self.w = nn.Conv2d(width, width, 1)

    def forward(self, x):
        x = self.conv(x) + self.w(x)
        return x if self.last else F.gelu(x)


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int, modes3: int):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))

    def compl_mul3d(self, inputs, weights):
        return torch.einsum("bixyz,ioxyz->boxyz", inputs, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        out_ft = torch.zeros(
            batchsize,
            self.weights1.shape[1],
            x.size(-3),
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, : self.modes2, : self.modes3],
            self.weights1,
        )
        out_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3],
            self.weights2,
        )
        out_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3],
            self.weights3,
        )
        out_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3],
            self.weights4,
        )
        return torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))


class FNO3DLayer(nn.Module):
    def __init__(self, modes1: int, modes2: int, modes3: int, width: int, *, last: bool = False):
        super().__init__()
        self.last = last
        self.conv = SpectralConv3d(width, width, modes1, modes2, modes3)
        self.w = nn.Conv3d(width, width, 1)

    def forward(self, x):
        x = self.conv(x) + self.w(x)
        return x if self.last else F.gelu(x)
