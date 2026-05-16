from __future__ import annotations

import numpy as np


class SPDE:
    """Small non-singular SPDE helper used to define NORS integration maps.

    This mirrors the DLR API names used by Graph.create_model_graph*. It only
    implements the parabolic tree integrations needed for MFV construction.
    """

    def __init__(self, BC="P", eps=1.0, T=None, X=None, Y=None):
        self.BC = BC
        self.eps = eps
        self.T = T
        self.X = X
        self.Y = Y

    def _parabolic_matrix(self, n_points, dt, dx, inverse=True):
        n = n_points - 1
        mat = np.diag(-2 * np.ones(n + 1)) + np.diag(np.ones(n), k=1) + np.diag(np.ones(n), k=-1)
        if self.BC == "D":
            mat[0, 0], mat[0, 1], mat[1, 0] = 0, 0, 0
            mat[-1, -1], mat[-1, -2], mat[-2, -1] = 0, 0, 0
        elif self.BC == "N":
            mat[0, 1], mat[-1, -2] = 2, 2
        elif self.BC == "P":
            mat[-1, 1], mat[0, -2] = 1, 1
        else:
            raise ValueError(f"Unknown boundary condition: {self.BC}")
        scaled = self.eps * dt * mat / (dx**2)
        return np.linalg.inv(np.eye(n + 1) - scaled) if inverse else scaled

    def _heat_integrate_1d(self, source):
        T = np.asarray(self.T)
        X = np.asarray(self.X)
        dt = float(T[1] - T[0])
        dx = float(X[1] - X[0])
        mat = self._parabolic_matrix(len(X), dt, dx).T
        out = np.zeros_like(source)
        for i in range(1, len(T)):
            out[:, i] = (out[:, i - 1] + source[:, i] * dt) @ mat
        return out

    def Integrate_Parabolic_trees(self, model, planted=None, exceptions=None, derivative=False):
        planted = set() if planted is None else set(planted)
        exceptions = set() if exceptions is None else set(exceptions)
        out = {}
        for tree, value in model.items():
            integrated_tree = f"I[{tree}]"
            if integrated_tree in planted or integrated_tree in exceptions:
                continue
            out[integrated_tree] = self._heat_integrate_1d(value)
        return out

    def _laplace_i_2d(self, arr, dt, dx):
        center = arr
        up = np.roll(arr, 1, axis=-2)
        down = np.roll(arr, -1, axis=-2)
        left = np.roll(arr, 1, axis=-1)
        right = np.roll(arr, -1, axis=-1)
        diag = (
            np.roll(up, 1, axis=-1)
            + np.roll(up, -1, axis=-1)
            + np.roll(down, 1, axis=-1)
            + np.roll(down, -1, axis=-1)
        )
        lap = 0.5 * (up + down + left + right) + 0.25 * diag - 3.0 * center
        return center + self.eps * dt * lap / (dx**2)

    def _heat_integrate_2d(self, source):
        T = np.asarray(self.T)
        X = np.asarray(self.X)
        if X.ndim == 2:
            x_line = X[:, 0]
        else:
            x_line = X
        dt = float(T[1] - T[0])
        dx = float(x_line[1] - x_line[0])
        out = np.zeros_like(source)
        for i in range(1, len(T)):
            out[:, i] = self._laplace_i_2d(out[:, i - 1], dt, dx) + source[:, i] * dt
        return out

    def Integrate_Parabolic_trees_2d(self, model, planted=None, exceptions=None, derivative=False):
        planted = set() if planted is None else set(planted)
        exceptions = set() if exceptions is None else set(exceptions)
        out = {}
        for tree, value in model.items():
            integrated_tree = f"I[{tree}]"
            if integrated_tree in planted or integrated_tree in exceptions:
                continue
            out[integrated_tree] = self._heat_integrate_2d(value)
        return out
