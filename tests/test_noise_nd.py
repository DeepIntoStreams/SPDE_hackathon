import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_gen.src.NoiseND import NoiseND


def test_partition_nd_matches_axis_shape():
    noise = NoiseND()

    x, y, z = noise.partition_nd(((0.0, 1.0), (-1.0, 1.0), (2.0, 3.0)), (0.25, 1.0, 0.5))

    assert x.shape == (5,)
    assert y.shape == (3,)
    assert z.shape == (3,)


def test_wiener_noise_shapes_for_1d_2d_3d():
    noise = NoiseND()
    np.random.seed(0)

    w1 = noise.WN_space_time(
        0.0, 0.2, 0.1, bounds=((0.0, 1.0),), steps=(0.5,), truncation=(4,)
    )
    w2 = noise.WN_space_time(
        0.0,
        0.2,
        0.1,
        bounds=((0.0, 1.0), (0.0, 1.0)),
        steps=(0.5, 0.5),
        truncation=(4, 3),
    )
    w3 = noise.WN_space_time(
        0.0,
        0.2,
        0.1,
        bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        steps=(0.5, 0.5, 0.5),
        truncation=(2, 3, 4),
        num=2,
    )

    assert w1.shape == (3, 3)
    assert w2.shape == (3, 3, 3)
    assert w3.shape == (2, 3, 3, 3, 3)
    assert np.allclose(w1[0], 0.0)
    assert np.allclose(w2[0], 0.0)
    assert np.allclose(w3[:, 0], 0.0)


def test_q_wiener_scales_modes_by_spectrum():
    cylindrical = NoiseND()
    q_wiener = NoiseND(
        covariance="q_wiener",
        q_spectrum=lambda modes, lengths: 0.25 if modes == (1,) else 1.0,
    )
    x = cylindrical.partition_axis(-1.0, 2.0, 1.0)

    basis_cyl = cylindrical._spatial_basis((x,), (3,), (3.0,))
    basis_q = q_wiener._spatial_basis((x,), (3,), (3.0,))

    assert np.allclose(basis_q[0], 0.5 * basis_cyl[0])
    assert np.allclose(basis_q[1:], basis_cyl[1:])


def test_sin_basis_vanishes_on_interval_endpoints():
    noise = NoiseND(basis="sin")
    x = noise.partition_axis(-1.0, 2.0, 1.0)
    basis = noise._spatial_basis((x,), (3,), (3.0,))

    assert basis.shape == (3, 4)
    assert np.allclose(basis[:, 0], 0.0)
    assert np.allclose(basis[:, -1], 0.0)
    assert not np.allclose(basis[0], 0.0)


def test_initial_shapes_and_periodic_endpoint():
    noise = NoiseND()
    x = noise.partition_axis(-0.5, 0.5, 0.25)
    y = noise.partition_axis(0.0, 1.0, 0.5)
    z = noise.partition_axis(0.0, 1.0, 0.5)

    np.random.seed(1)
    init_1d = noise.initial(4, (x,), truncation=(3,))
    init_3d = noise.initial(2, (x, y, z), truncation=(2, 2, 2))

    assert init_1d.shape == (4, 5)
    assert init_3d.shape == (2, 5, 3, 3)
    assert np.allclose(init_1d[:, 0], init_1d[:, -1])
