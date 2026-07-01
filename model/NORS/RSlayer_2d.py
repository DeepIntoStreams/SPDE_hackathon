from __future__ import annotations

from model.NORS.layers import FNO3DLayer as FNO_layer
from model.NORS.layers import ParabolicIntegrate2D


class ParabolicIntegrate_2d(ParabolicIntegrate2D):
    """DLR-compatible 2D NORS MFV layer name."""

    def __init__(self, graph, T, X, Y, BC="P", eps=1, device=None, dtype=None):
        super().__init__(graph, T, X, Y, eps=eps)
