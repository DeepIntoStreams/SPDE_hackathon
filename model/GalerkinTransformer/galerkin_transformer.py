"""
Galerkin Transformer for SPDE learning (1D and 2D, time-dependent).

Adapted from the original Galerkin Transformer for operator learning:
    Cao, S. (2021). "Choose a Transformer: Fourier or Galerkin."
    NeurIPS 2021. https://arxiv.org/abs/2105.14995
    Code: https://github.com/scaomath/galerkin-transformer
    License: MIT

Modifications for SPDEBench:
  - Added support for time-dependent operator learning (xi -> u and (u0, xi) -> u)
  - Added 2D spatial support via factorised attention over (x, y) axes
  - Adapted input/output interface to match existing SPDEBench dataloaders
  - Self-contained implementation (no external galerkin-transformer package needed)
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


#attention mechanism: Galerkin (linear, softmax-free) attention

class GalerkinAttention(nn.Module):
    """
    Galerkin-type (global) linear attention without softmax normalisation.

    The key insight from Cao (2021) is that removing softmax and applying
    instance normalisation to K and V instead yields a Petrov-Galerkin
    projection that is more suitable for operator learning than standard
    scaled dot-product attention.

    Complexity: O(n * d^2) instead of O(n^2 * d), where n = sequence length.

    References:
        Cao (2021), Theorem 3.1: the linear attention variant approximates
        a Petrov-Galerkin projection in a Hilbert space.
    """

    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # Instance normalisation applied column-wise to K and V,
        # mimicking the Gram-Schmidt step in the Galerkin method.
        # norm over (seq_len,) for each (batch, head, d_head) triplet.
        self.norm_k = nn.InstanceNorm1d(d_model)
        self.norm_v = nn.InstanceNorm1d(d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        # Xavier with small diagonal weight as in Cao (2021) §4
        xavier_init = 0.01
        diagonal_weight = 0.01
        for w in [self.W_q, self.W_k, self.W_v]:
            nn.init.xavier_uniform_(w.weight, gain=xavier_init)
            w.weight.data += diagonal_weight * torch.diag(
                torch.ones(w.weight.size(-1),
                           device=w.weight.device,
                           dtype=w.weight.dtype)
            )
            nn.init.zeros_(w.bias)
        nn.init.xavier_uniform_(self.W_o.weight)
        nn.init.zeros_(self.W_o.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n, d_model)  — sequence of node embeddings
        Returns:
            out: (B, n, d_model)
        """
        B, n, _ = x.shape

        Q = self.W_q(x)  # (B, n, d_model)
        # Apply instance normalisation on K and V before projecting to heads
        K = self.norm_k(self.W_k(x).transpose(1, 2)).transpose(1, 2)  # (B, n, d_model) — norm over n
        V = self.norm_v(self.W_v(x).transpose(1, 2)).transpose(1, 2)

        # Reshape to (B, n_head, n, d_head)
        def split_heads(t):
            return t.view(B, n, self.n_head, self.d_head).transpose(1, 2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # Galerkin linear attention: V K^T Q in O(n d^2)
        # Compute K^T V: (B, n_head, d_head, d_head)
        KtV = torch.einsum('bhnd,bhnm->bhdm', K, V) / n
        # Apply to Q: (B, n_head, n, d_head)
        out = torch.einsum('bhnd,bhdm->bhnm', Q, KtV)

        out = self.dropout(out)
        #merge heads
        out = out.transpose(1, 2).contiguous().view(B, n, self.d_model)
        return self.W_o(out)


#transformer encoder layer

class GalerkinEncoderLayer(nn.Module):
    """Single Galerkin Transformer encoder layer (Pre-LN variant)."""

    def __init__(self, d_model: int, n_head: int,
                 dim_feedforward: int = 512, dropout: float = 0.0):
        super().__init__()
        self.attn = GalerkinAttention(d_model, n_head, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


# 1D Galerkin Transformer for SPDE learning (Burgers, KdV, etc.)

class GalerkinTransformer1D(nn.Module):
    """
    Galerkin Transformer for 1D time-dependent SPDE learning.

    Supports two tasks:
        xi_to_u    : xi(t, x)         -> u(t, x)  [task='xi']
        u0xi_to_u  : (u0(x), xi(t,x)) -> u(t, x)  [task='u0xi']

    Architecture:
        1. Lift input to d_model with a pointwise linear layer.
        2. Concatenate spatial coordinate embedding.
        3. L Galerkin encoder layers operating over the (T*Nx) sequence.
        4. Project to output dimension pointwise.

    The (x, t) coordinates are appended as positional features before
    lifting, following the approach of Cao (2021) §5.
    """

    def __init__(
        self,
        T: int = 51,
        dim_x: int = 128,
        d_model: int = 64,
        n_head: int = 4,
        n_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        task: str = 'xi',         # 'xi' or 'u0xi'
    ):
        super().__init__()
        self.T = T
        self.dim_x = dim_x
        self.task = task

        # input channels: noise xi has 1 channel; u0 adds 1 more if u0xi task
        in_channels = 1 + (1 if task == 'u0xi' else 0)
        # positional features: (x_coord, t_coord) → 2 dims
        in_channels_with_pos = in_channels + 2

        self.lifting = nn.Linear(in_channels_with_pos, d_model)

        self.encoder = nn.ModuleList([
            GalerkinEncoderLayer(d_model, n_head, dim_feedforward, dropout)
            for _ in range(n_layers)
        ])

        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        #precompute coordinate grid (registered as buffer so it moves with .to(device))
        x_grid = torch.linspace(0, 1, dim_x)           # (Nx,)
        t_grid = torch.linspace(0, 1, T)                # (T,)
        grid_t, grid_x = torch.meshgrid(t_grid, x_grid, indexing='ij')  # (T, Nx)
        # flatten to sequence: (T*Nx, 2)
        coords = torch.stack([grid_x.flatten(), grid_t.flatten()], dim=-1)
        self.register_buffer('coords', coords)  # (T*Nx, 2)

    def forward(self, xi: torch.Tensor,
                u0: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            xi: (B, T, Nx)     — noise path
            u0: (B, Nx)        — initial condition (only for task='u0xi')
        Returns:
            u:  (B, T, Nx)     — predicted solution
        """
        B, T, Nx = xi.shape

        #reshape xi to (B, T*Nx, 1)
        x_in = xi.reshape(B, T * Nx, 1)

        if self.task == 'u0xi':
            assert u0 is not None, "u0 required for task='u0xi'"
            #broadcast u0 over time: (B, Nx) -> (B, T, Nx) -> (B, T*Nx, 1)
            u0_exp = u0.unsqueeze(1).expand(B, T, Nx).reshape(B, T * Nx, 1)
            x_in = torch.cat([x_in, u0_exp], dim=-1)  # (B, T*Nx, 2)

        # Append coordinate features: (B, T*Nx, 2)
        coords = self.coords.unsqueeze(0).expand(B, -1, -1)  # (B, T*Nx, 2)
        x_in = torch.cat([x_in, coords], dim=-1)  # (B, T*Nx, in_channels+2)

        # Lift
        h = self.lifting(x_in)  # (B, T*Nx, d_model)

        # Galerkin encoder
        for layer in self.encoder:
            h = layer(h)

        # Project to output
        out = self.projection(h)  # (B, T*Nx, 1)
        out = out.squeeze(-1).reshape(B, T, Nx)  # (B, T, Nx)
        return out


# 2D Galerkin Transformer for SPDE learning (NSE)


class GalerkinTransformer2D(nn.Module):
    """
    Galerkin Transformer for 2D time-dependent SPDE learning.

    For 2D spatial fields the full (T*Nx*Ny) sequence length is too large
    for attention (e.g. 100*16*16 = 25600 tokens). We use a factorised
    scheme: attention alternates between the spatial-x axis and spatial-y
    axis, following the FactFormer strategy but with Galerkin attention.

    Each attention layer processes:
        - x-attention:  reshape to (B*T*Ny, Nx, d)  → attend over Nx
        - y-attention:  reshape to (B*T*Nx, Ny, d)  → attend over Ny

    This keeps memory O(B * T * max(Nx, Ny) * d) instead of O(B*T*Nx*Ny*d).

    Supports tasks 'xi' and 'u0xi' with the same interface as 2D NSE in
    the existing SPDEBench FNO2D baseline.
    """

    def __init__(
        self,
        T: int = 101,
        dim_x: int = 16,
        dim_y: int = 16,
        d_model: int = 64,
        n_head: int = 4,
        n_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        task: str = 'xi',
    ):
        super().__init__()
        self.T = T
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.task = task

        in_channels = 1 + (1 if task == 'u0xi' else 0)
        #positional: (x, y, t) → 3 dims
        in_channels_with_pos = in_channels + 3

        self.lifting = nn.Linear(in_channels_with_pos, d_model)

        #alternating x/y attention layers
        self.x_layers = nn.ModuleList([
            GalerkinEncoderLayer(d_model, n_head, dim_feedforward, dropout)
            for _ in range(n_layers)
        ])
        self.y_layers = nn.ModuleList([
            GalerkinEncoderLayer(d_model, n_head, dim_feedforward, dropout)
            for _ in range(n_layers)
        ])

        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        # Pre-compute 3D coordinate grid
        x_g = torch.linspace(0, 1, dim_x)
        y_g = torch.linspace(0, 1, dim_y)
        t_g = torch.linspace(0, 1, T)
        gt, gx, gy = torch.meshgrid(t_g, x_g, y_g, indexing='ij')  # (T, Nx, Ny)
        coords = torch.stack([gx, gy, gt], dim=-1)  # (T, Nx, Ny, 3)
        self.register_buffer('coords', coords)

    def forward(self, xi: torch.Tensor,
                u0: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            xi: (B, T, Nx, Ny)    — noise path
            u0: (B, Nx, Ny)       — initial condition (task='u0xi' only)
        Returns:
            u:  (B, T, Nx, Ny)    — predicted solution
        """
        B, T, Nx, Ny = xi.shape

        #(B, T, Nx, Ny, 1)
        x_in = xi.unsqueeze(-1)

        if self.task == 'u0xi':
            assert u0 is not None
            u0_exp = u0.unsqueeze(1).expand(B, T, Nx, Ny).unsqueeze(-1)
            x_in = torch.cat([x_in, u0_exp], dim=-1)

        #Append coordinates: (B, T, Nx, Ny, 3)
        coords = self.coords.unsqueeze(0).expand(B, -1, -1, -1, -1)
        x_in = torch.cat([x_in, coords], dim=-1)  # (B, T, Nx, Ny, in_ch+3)

        #lift: (B, T, Nx, Ny, d_model)
        h = self.lifting(x_in)

        for x_layer, y_layer in zip(self.x_layers, self.y_layers):
            # x-attention: fold (B, T, Ny) into batch → (B*T*Ny, Nx, d)
            Bv = B * T * Ny
            hx = h.permute(0, 1, 3, 2, 4).reshape(Bv, Nx, -1)
            hx = x_layer(hx)
            h = hx.reshape(B, T, Ny, Nx, -1).permute(0, 1, 3, 2, 4)

            # y-attention: fold (B, T, Nx) into batch → (B*T*Nx, Ny, d)
            Bv = B * T * Nx
            hy = h.reshape(Bv, Ny, -1)
            hy = y_layer(hy)
            h = hy.reshape(B, T, Nx, Ny, -1)

        out = self.projection(h).squeeze(-1)  # (B, T, Nx, Ny)
        return out