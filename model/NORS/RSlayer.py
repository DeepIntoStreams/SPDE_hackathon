from __future__ import annotations

from model.NORS.layers import FNO2DLayer as FNO_layer
from model.NORS.layers import ParabolicIntegrate1D


class ParabolicIntegrate(ParabolicIntegrate1D):
    """DLR-compatible 1D NORS MFV layer name."""

    def __init__(self, graph, BC="P", eps=1, T=None, X=None, device=None, dtype=None):
        super().__init__(graph, T, X, bc=BC, eps=eps)
