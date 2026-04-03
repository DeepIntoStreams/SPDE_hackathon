import string

import numpy as np
from tqdm import tqdm


class NoiseND:
    def partition_axis(self, a, b, step):
        return np.linspace(a, b, int((b - a) / step) + 1)

    def partition_nd(self, bounds, steps):
        bounds = tuple(bounds)
        steps = tuple(steps)
        if len(bounds) != len(steps):
            raise ValueError("bounds and steps must have the same length")

        return tuple(
            self.partition_axis(start, stop, step)
            for (start, stop), step in zip(bounds, steps)
        )

    def brownian_motion(self, start, stop, dt, mode_shape):
        mode_shape = tuple(int(m) for m in mode_shape)
        n_steps = int((stop - start) / dt) + 1
        increments = np.random.normal(
            scale=np.sqrt(dt), size=(n_steps, int(np.prod(mode_shape)))
        )
        increments[0] = 0
        return np.cumsum(increments, axis=0)

    def WN_space_time_single(self, s, t, dt, bounds, steps, truncation, correlation=None):
        steps = tuple(float(step) for step in steps)
        grids = self.partition_nd(bounds, steps)
        mode_shape = self._normalize_truncation(truncation, len(grids))
        lengths = tuple(step * (len(grid) - 1) for grid, step in zip(grids, steps))
        basis = self._build_spatial_basis(grids, mode_shape, lengths, correlation)
        brownian = self.brownian_motion(s, t, dt, mode_shape)
        return np.einsum("tm,m...->t...", brownian, basis, optimize=True)

    def WN_space_time_many(
        self, s, t, dt, bounds, steps, num, truncation, correlation=None
    ):
        steps = tuple(float(step) for step in steps)
        grids = self.partition_nd(bounds, steps)
        mode_shape = self._normalize_truncation(truncation, len(grids))
        lengths = tuple(step * (len(grid) - 1) for grid, step in zip(grids, steps))
        basis = self._build_spatial_basis(grids, mode_shape, lengths, correlation)
        return np.array(
            [
                np.einsum(
                    "tm,m...->t...",
                    self.brownian_motion(s, t, dt, mode_shape),
                    basis,
                    optimize=True,
                )
                for _ in tqdm(range(num))
            ]
        )

    def initial(self, num, grids, truncation=10, decay=2, scaling=1, dirichlet=False):
        grids = tuple(np.asarray(grid) for grid in grids)
        dim = len(grids)
        mode_limits = self._normalize_truncation(truncation, dim)
        grid_shape = tuple(len(grid) for grid in grids)
        meshes = np.meshgrid(*grids, indexing="ij")
        scales = tuple(self._grid_scale(grid, scaling) for grid in grids)
        mode_ranges = [np.arange(-limit, limit + 1) for limit in mode_limits]
        basis = np.empty(tuple(len(rng) for rng in mode_ranges) + grid_shape)
        signs = tuple(-1 if axis % 2 else 1 for axis in range(dim))

        for basis_index in np.ndindex(*(len(rng) for rng in mode_ranges)):
            modes = tuple(mode_ranges[axis][index] for axis, index in enumerate(basis_index))
            phase = np.zeros(grid_shape)
            for mesh, scale, mode, sign in zip(meshes, scales, modes, signs):
                phase += sign * mode * np.pi * mesh / (scale * dim)
            weight = 1 + sum(np.abs(mode) ** decay for mode in modes)
            basis[basis_index] = np.sin(phase) / weight

        basis_flat = basis.reshape(int(np.prod([len(rng) for rng in mode_ranges])), *grid_shape)

        all_coeffs = np.random.normal(size=(num, basis_flat.shape[0]))
        initial_conditions = np.einsum("nm,m...->n...", all_coeffs, basis_flat, optimize=True)
        if not dirichlet:
            initial_conditions += np.random.normal(size=(num,) + (1,) * dim)
        if dim == 1:
            initial_conditions[:, -1] = initial_conditions[:, 0]

        return initial_conditions

    def _build_spatial_basis(self, grids, mode_shape, lengths, correlation):
        grid_shape = tuple(len(grid) for grid in grids)
        n_modes = int(np.prod(mode_shape))

        if correlation is None:
            factors = []
            for grid, n_modes_axis, length in zip(grids, mode_shape, lengths):
                modes = np.arange(n_modes_axis, dtype=float)
                factors.append(np.sin(np.pi * np.outer(modes, grid) / length))

            mode_labels = self._axis_labels("m", len(grids))
            coord_labels = self._axis_labels("x", len(grids))
            equation = ",".join(
                f"{mode_label}{coord_label}"
                for mode_label, coord_label in zip(mode_labels, coord_labels)
            )
            equation += "->" + "".join(mode_labels + coord_labels)
            scale = np.sqrt((2 ** len(lengths)) / np.prod(lengths))
            basis = scale * np.einsum(equation, *factors, optimize=True)
            return basis.reshape(n_modes, *grid_shape)

        coord_meshes = np.meshgrid(*grids, indexing="ij")
        basis = np.empty((n_modes,) + grid_shape)
        for flat_index, modes in enumerate(np.ndindex(mode_shape)):
            basis[flat_index] = correlation(coord_meshes, modes, lengths)
        return basis

    def _grid_scale(self, grid, scaling):
        scale = float(np.max(grid)) / scaling if scaling else float(np.max(grid))
        return scale if scale != 0 else 1.0

    def _normalize_truncation(self, truncation, dim):
        if np.isscalar(truncation):
            values = (int(truncation),) * dim
        else:
            values = tuple(int(value) for value in truncation)

        if len(values) != dim:
            raise ValueError("truncation must provide one value per spatial dimension")
        if any(value < 1 for value in values):
            raise ValueError("truncation values must be positive")
        return values

    def _axis_labels(self, prefix, dim):
        start = string.ascii_lowercase.index(prefix)
        if start + dim > 26:
            raise ValueError(
                f"Too many spatial dimensions ({dim}) for einsum labels with prefix '{prefix}'"
            )
        return list(string.ascii_lowercase[start:start + dim])
