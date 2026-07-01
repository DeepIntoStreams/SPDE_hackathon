from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from model.NORS.layers import FNO2DLayer, FNO3DLayer


class NORS1D(nn.Module):
    def __init__(
        self,
        num_tree: int,
        *,
        modes_x: int = 16,
        modes_t: int = 16,
        width: int = 16,
        layers: int = 4,
        padding: int = 6,
    ):
        super().__init__()
        self.padding = padding
        self.num_tree = num_tree
        self.fc0 = nn.Linear(self.num_tree + 2, width)
        blocks = [FNO2DLayer(modes_x, modes_t, width) for _ in range(layers - 1)]
        blocks.append(FNO2DLayer(modes_x, modes_t, width, last=True))
        self.net = nn.Sequential(*blocks)
        self.decoder = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 1))

    @staticmethod
    def get_grid(batch: int, nt: int, nx: int, device) -> torch.Tensor:
        grid_t = torch.linspace(0, 1, nt, device=device).reshape(1, nt, 1, 1).expand(batch, nt, nx, 1)
        grid_x = torch.linspace(0, 1, nx, device=device).reshape(1, 1, nx, 1).expand(batch, nt, nx, 1)
        return torch.cat((grid_x, grid_t), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: precomputed MFV with shape [B, T, X, C_tree]."""
        grid = self.get_grid(x.shape[0], x.shape[1], x.shape[2], x.device)
        x = torch.cat((x, grid), dim=-1)
        x = x.to(torch.float32)
        x = self.fc0(x)
        x = x.permute(0, 3, 2, 1)  # [B, C, X, T]
        x = F.pad(x, [0, self.padding])
        x = self.net(x)
        x = x[..., : -self.padding]
        x = x.permute(0, 3, 2, 1)
        return self.decoder(x).squeeze(-1)


class NORS2D(nn.Module):
    def __init__(
        self,
        num_tree: int,
        *,
        modes_x: int = 8,
        modes_y: int = 8,
        modes_t: int = 8,
        width: int = 16,
        layers: int = 4,
        padding: int = 6,
    ):
        super().__init__()
        self.padding = padding
        self.num_tree = num_tree
        self.fc0 = nn.Linear(self.num_tree + 3, width)
        blocks = [FNO3DLayer(modes_x, modes_y, modes_t, width) for _ in range(layers - 1)]
        blocks.append(FNO3DLayer(modes_x, modes_y, modes_t, width, last=True))
        self.net = nn.Sequential(*blocks)
        self.decoder = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 1))

    @staticmethod
    def get_grid(batch: int, nt: int, nx: int, ny: int, device) -> torch.Tensor:
        grid_t = torch.linspace(0, 1, nt, device=device).reshape(1, nt, 1, 1, 1).expand(batch, nt, nx, ny, 1)
        grid_x = torch.linspace(0, 1, nx, device=device).reshape(1, 1, nx, 1, 1).expand(batch, nt, nx, ny, 1)
        grid_y = torch.linspace(0, 1, ny, device=device).reshape(1, 1, 1, ny, 1).expand(batch, nt, nx, ny, 1)
        return torch.cat((grid_x, grid_y, grid_t), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: precomputed MFV with shape [B, T, X, Y, C_tree]."""
        grid = self.get_grid(
            x.shape[0],
            x.shape[1],
            x.shape[2],
            x.shape[3],
            x.device,
        )
        x = torch.cat((x, grid), dim=-1)
        x = x.to(torch.float32)
        x = self.fc0(x)
        x = x.permute(0, 4, 2, 3, 1)  # [B, C, X, Y, T]
        x = F.pad(x, [0, self.padding])
        x = self.net(x)
        x = x[..., : -self.padding]
        x = x.permute(0, 4, 2, 3, 1)
        return self.decoder(x).squeeze(-1)
