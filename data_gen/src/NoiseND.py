import string
from typing import Callable, Literal

import numpy as np
from tqdm import tqdm

Basis1D = Callable[[np.ndarray, int, float], np.ndarray]
SpectrumFn = Callable[[tuple[int, ...], tuple[float, ...]], float]


def _sin_basis(grid: np.ndarray, n_modes: int, length: float) -> np.ndarray:
    """Dirichlet: sqrt(2/L) sin(k pi (x-x0) / L), k = 1..N."""
    k = np.arange(1, n_modes + 1, dtype=float)
    shifted = grid - grid[0]
    return np.sqrt(2.0 / length) * np.sin(np.pi * np.outer(k, shifted) / length)


def _sincos_basis(grid: np.ndarray, n_modes: int, length: float) -> np.ndarray:
    """Periodic (real Fourier): [1, cos_1, sin_1, cos_2, sin_2, ...] truncated to N modes."""
    shifted = grid - grid[0]
    basis = np.empty((n_modes, grid.size))
    for k in range(n_modes):
        if k == 0:
            basis[k] = 1.0 / np.sqrt(length)
            continue
        harmonic = (k + 1) // 2
        angle = 2.0 * np.pi * harmonic * shifted / length
        basis[k] = np.sqrt(2.0 / length) * (np.cos(angle) if k % 2 == 1 else np.sin(angle))
    return basis


def _fourier_basis(grid: np.ndarray, n_modes: int, length: float) -> np.ndarray:
    """Periodic (complex Fourier): (1/sqrt(L)) exp(i 2 pi n (x-x0) / L), n centred on 0."""
    shifted = grid - grid[0]
    start = -(n_modes // 2)
    n = np.arange(start, start + n_modes, dtype=float)
    return np.exp(1j * 2.0 * np.pi * np.outer(n, shifted) / length) / np.sqrt(length)


_BASIS_FNS: dict[str, Basis1D] = {
    "sin": _sin_basis,
    "sincos": _sincos_basis,
    "fourier": _fourier_basis,
}


class NoiseND:
    """N-dimensional SPDE noise from tensor products of 1D spectral bases."""

    def __init__(
        self,
        basis: Literal["sin", "sincos", "fourier"] = "sincos",
        covariance: Literal["cylindrical", "q_wiener"] = "cylindrical",
        q_spectrum: SpectrumFn | None = None,
    ):
        if basis not in _BASIS_FNS:
            raise ValueError(f"basis must be one of {list(_BASIS_FNS)}; got {basis!r}")
        if covariance not in {"cylindrical", "q_wiener"}:
            raise ValueError("covariance must be 'cylindrical' or 'q_wiener'")
        if covariance == "q_wiener" and q_spectrum is None:
            raise ValueError("q_spectrum is required when covariance='q_wiener'")
        if covariance == "cylindrical" and q_spectrum is not None:
            raise ValueError("q_spectrum is only valid when covariance='q_wiener'")
        self.basis = basis
        self.covariance = covariance
        self._basis_1d = _BASIS_FNS[basis]
        self._q_spectrum = q_spectrum

    @staticmethod
    def partition_axis(a: float, b: float, step: float) -> np.ndarray:
        return np.linspace(a, b, int((b - a) / step) + 1)

    def partition_nd(self, bounds, steps) -> tuple[np.ndarray, ...]:
        if len(bounds) != len(steps):
            raise ValueError("bounds and steps must have the same length")
        return tuple(self.partition_axis(a, b, s) for (a, b), s in zip(bounds, steps))

    def WN_space_time(self, s, t, dt, bounds, steps, truncation, num=None):
        grids = self.partition_nd(bounds, steps)
        mode_shape = self._normalize_truncation(truncation, len(grids))
        lengths = tuple(float(step) * (g.size - 1) for g, step in zip(grids, steps))
        basis = self._spatial_basis(grids, mode_shape, lengths)
        total_modes = int(np.prod(mode_shape))

        if num is None:
            bm = self._brownian(s, t, dt, total_modes)
            return np.einsum("tm,m...->t...", bm, basis, optimize=True)

        return np.stack(
            [
                np.einsum(
                    "tm,m...->t...",
                    self._brownian(s, t, dt, total_modes),
                    basis,
                    optimize=True,
                )
                for _ in tqdm(range(num))
            ]
        )

    def initial(self, num, grids, truncation=10, decay=2):
        grids = tuple(np.asarray(g, dtype=float) for g in grids)
        mode_shape = self._normalize_truncation(truncation, len(grids))
        lengths = tuple(float(g[-1] - g[0]) for g in grids)
        basis = self._spatial_basis(grids, mode_shape, lengths)
        total_modes = int(np.prod(mode_shape))

        idx = np.indices(mode_shape).reshape(len(mode_shape), -1) + 1
        weights = 1.0 / (1.0 + np.sum(idx.astype(float) ** decay, axis=0))
        coeffs = np.random.normal(size=(num, total_modes)) * weights
        return np.einsum("nm,m...->n...", coeffs, basis, optimize=True)

    def _spatial_basis(self, grids, mode_shape, lengths):
        factors = [
            self._basis_1d(np.asarray(g, dtype=float), n, L)
            for g, n, L in zip(grids, mode_shape, lengths)
        ]
        d = len(factors)
        if d > 13:
            raise ValueError(f"dimension {d} exceeds einsum label budget")
        modes = string.ascii_lowercase[:d]
        coords = string.ascii_lowercase[13 : 13 + d]
        eq = ",".join(m + c for m, c in zip(modes, coords)) + f"->{modes}{coords}"
        basis = np.einsum(eq, *factors, optimize=True)
        basis = basis.reshape(int(np.prod(mode_shape)), *[g.size for g in grids])
        mode_scales = self._mode_scales(mode_shape, lengths)
        return basis * mode_scales.reshape((-1,) + (1,) * len(grids))

    def _mode_scales(self, mode_shape, lengths):
        total_modes = int(np.prod(mode_shape))
        if self.covariance == "cylindrical":
            return np.ones(total_modes, dtype=float)

        scales = np.empty(total_modes, dtype=float)
        for flat_index, mode_index in enumerate(np.ndindex(mode_shape)):
            modes = tuple(index + 1 for index in mode_index)
            scales[flat_index] = np.sqrt(float(self._q_spectrum(modes, lengths)))
        return scales

    @staticmethod
    def _brownian(start, stop, dt, n_modes):
        n_steps = int((stop - start) / dt) + 1
        incr = np.random.normal(scale=np.sqrt(dt), size=(n_steps, n_modes))
        incr[0] = 0
        return np.cumsum(incr, axis=0)

    @staticmethod
    def _normalize_truncation(truncation, dim):
        values = (int(truncation),) * dim if np.isscalar(truncation) else tuple(int(v) for v in truncation)
        if len(values) != dim or any(v < 1 for v in values):
            raise ValueError("truncation must be a positive int or tuple of positive ints, one per spatial dimension")
        return values
