import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




class Conv4dPointwise(nn.Module):
    """Pointwise 4D convolution implemented with einsum."""

    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(in_channels) if in_channels > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        out = torch.einsum("bcxyzt,oc->boxyzt", x, self.weight)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1, 1, 1)
        return out


class SpectralConv4d(nn.Module):
    """4D Fourier convolution on (x, y, z, t)."""

    def __init__(self, in_channels, out_channels, modes_x, modes_y, modes_z, modes_t):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_z = modes_z
        self.modes_t = modes_t

        scale = 1 / (in_channels * out_channels)
        shape = (in_channels, out_channels, modes_x, modes_y, modes_z, modes_t)
        self.weights1 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.weights5 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.weights6 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.weights7 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.weights8 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))

    @staticmethod
    def complex_mul4d(x_ft, weights):
        return torch.einsum("bixyzt,ioxyzt->boxyzt", x_ft, weights)

    def forward(self, x):
        batch_size = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[-4, -3, -2, -1])

        mx = min(self.modes_x, x_ft.size(-4))
        my = min(self.modes_y, x_ft.size(-3))
        mz = min(self.modes_z, x_ft.size(-2))
        mt = min(self.modes_t, x_ft.size(-1))

        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            x.size(-4),
            x.size(-3),
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        if mx > 0 and my > 0 and mz > 0 and mt > 0:
            out_ft[:, :, :mx, :my, :mz, :mt] = self.complex_mul4d(
                x_ft[:, :, :mx, :my, :mz, :mt],
                self.weights1[:, :, :mx, :my, :mz, :mt],
            )

            if mx < x_ft.size(-4):
                out_ft[:, :, -mx:, :my, :mz, :mt] = self.complex_mul4d(
                    x_ft[:, :, -mx:, :my, :mz, :mt],
                    self.weights2[:, :, :mx, :my, :mz, :mt],
                )

            if my < x_ft.size(-3):
                out_ft[:, :, :mx, -my:, :mz, :mt] = self.complex_mul4d(
                    x_ft[:, :, :mx, -my:, :mz, :mt],
                    self.weights3[:, :, :mx, :my, :mz, :mt],
                )

            if mz < x_ft.size(-2):
                out_ft[:, :, :mx, :my, -mz:, :mt] = self.complex_mul4d(
                    x_ft[:, :, :mx, :my, -mz:, :mt],
                    self.weights4[:, :, :mx, :my, :mz, :mt],
                )

            if mx < x_ft.size(-4) and my < x_ft.size(-3):
                out_ft[:, :, -mx:, -my:, :mz, :mt] = self.complex_mul4d(
                    x_ft[:, :, -mx:, -my:, :mz, :mt],
                    self.weights5[:, :, :mx, :my, :mz, :mt],
                )

            if mx < x_ft.size(-4) and mz < x_ft.size(-2):
                out_ft[:, :, -mx:, :my, -mz:, :mt] = self.complex_mul4d(
                    x_ft[:, :, -mx:, :my, -mz:, :mt],
                    self.weights6[:, :, :mx, :my, :mz, :mt],
                )

            if my < x_ft.size(-3) and mz < x_ft.size(-2):
                out_ft[:, :, :mx, -my:, -mz:, :mt] = self.complex_mul4d(
                    x_ft[:, :, :mx, -my:, -mz:, :mt],
                    self.weights7[:, :, :mx, :my, :mz, :mt],
                )

            if mx < x_ft.size(-4) and my < x_ft.size(-3) and mz < x_ft.size(-2):
                out_ft[:, :, -mx:, -my:, -mz:, :mt] = self.complex_mul4d(
                    x_ft[:, :, -mx:, -my:, -mz:, :mt],
                    self.weights8[:, :, :mx, :my, :mz, :mt],
                )

        return torch.fft.irfftn(
            out_ft,
            s=(x.size(-4), x.size(-3), x.size(-2), x.size(-1)),
            dim=[-4, -3, -2, -1],
        )


class FNO4DLayer(nn.Module):
    """Single FNO block on (x, y, z, t)."""

    def __init__(self, modes_x, modes_y, modes_z, modes_t, width, last=False):
        super().__init__()
        self.last = last
        self.spectral = SpectralConv4d(width, width, modes_x, modes_y, modes_z, modes_t)
        self.pointwise = Conv4dPointwise(width, width, bias=True)

    def forward(self, x):
        x = self.spectral(x) + self.pointwise(x)
        if not self.last:
            x = F.gelu(x)
        return x


class FNO3D(nn.Module):
    """
    Plain FNO baseline for Phi43.

    Input:
        xi: [B, X, Y, Z, T, 6]
            channels are [dW, x, y, z, t, u0]
    Output:
        u: [B, X, Y, Z, T]
    """

    def __init__(
        self,
        modes_x=8,
        modes_y=8,
        modes_z=8,
        modes_t=12,
        width=32,
        n_layers=3,
        dropout=0.05,
        padding=6,
    ):
        super().__init__()
        self.padding = padding
        self.fc0 = nn.Linear(6, width)
        self.layers = nn.ModuleList(
            [
                FNO4DLayer(
                    modes_x=modes_x,
                    modes_y=modes_y,
                    modes_z=modes_z,
                    modes_t=modes_t,
                    width=width,
                    last=(i == n_layers - 1),
                )
                for i in range(n_layers)
            ]
        )
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, xi):
        if xi.shape[-1] != 6:
            raise ValueError(f"Expected 6 input channels [dW,x,y,z,t,u0], got {xi.shape[-1]}")

        x = self.fc0(xi.to(dtype=torch.float32))
        x = x.permute(0, 5, 1, 2, 3, 4)
        if self.padding > 0:
            x = F.pad(x, [0, self.padding])

        bsz, channels, nx, ny, nz, nt = x.shape
        x_tmp = x.permute(0, 1, 5, 2, 3, 4).reshape(bsz, channels * nt, nx, ny, nz)
        x_tmp = self.dropout(x_tmp)
        x = x_tmp.reshape(bsz, channels, nt, nx, ny, nz).permute(0, 1, 3, 4, 5, 2)

        for layer in self.layers:
            x = layer(x)

        if self.padding > 0:
            x = x[..., :-self.padding]
        x = x.permute(0, 2, 3, 4, 5, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(-1)
