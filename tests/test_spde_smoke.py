import os
import sys
from types import SimpleNamespace

import numpy as np
import scipy.io
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_gen.examples.gen_KPZ import simulator as kpz_simulator
from data_gen.examples.gen_KdV import simulator as kdv_simulator
from data_gen.examples.gen_KdV_refine import simulator as kdv_refine_simulator
from data_gen.examples.gen_navier_stokes import simulator as ns_simulator
from data_gen.examples.gen_phi41 import simulator as phi41_simulator
from data_gen.examples.gen_phi42 import solver as phi42_simulator
from data_gen.examples.gen_phi43 import build_save_dict as phi43_build_save_dict
from data_gen.examples.gen_phi43 import simulator as phi43_simulator
from data_gen.examples.gen_wave import simulator as wave_simulator


def test_phi41_smoke():
    np.random.seed(0)
    x, t, w, sol = phi41_simulator(
        a=0.0, b=1.0, Nx=4, s=0.0, t=0.02, Nt=4, truncation=4, sigma=0.1, fix_u0=True, num=2
    )

    assert x.shape == (5,)
    assert t.shape == (5,)
    assert w.shape == (2, 5, 5)
    assert sol.shape == (2, 5, 5)


def test_phi42_smoke():
    np.random.seed(0)
    x, y, t, w, eps, sol_reno, sol_expl = phi42_simulator(
        a=0.0,
        b=1.0,
        Nx=2,
        c=0.0,
        d=1.0,
        Ny=2,
        s=0.0,
        t=0.002,
        Nt=2,
        num=1,
        eps=4,
        sigma=0.1,
        fix_u0=True,
    )

    assert x.shape == (3,)
    assert y.shape == (3,)
    assert t.shape == (3,)
    assert eps == 4
    assert w.shape == (1, 3, 3, 3)
    assert sol_reno.shape == (1, 3, 3, 3)
    assert sol_expl.shape == (1, 3, 3, 3)


def test_phi43_smoke(tmp_path):
    np.random.seed(0)
    torch.manual_seed(0)

    x, y, z, t, w, sol, spde, pre = phi43_simulator(
        N=1,
        dt=0.01,
        steps=2,
        num=1,
        fix_u0=True,
        num_tau=8,
        tau_max_multiplier=4.0,
        include_c12=True,
        seed=0,
    )

    assert x.shape == (3,)
    assert y.shape == (3,)
    assert z.shape == (3,)
    assert t.shape == (3,)
    assert w.shape == (1, 3, 3, 3, 3)
    assert sol.shape == (1, 3, 3, 3, 3)

    save_path = tmp_path / "phi43_smoke.mat"
    scipy.io.savemat(
        save_path,
        phi43_build_save_dict(
            x=x,
            y=y,
            z=z,
            t=t,
            w=w,
            sol=sol,
            spde=spde,
            pre=pre,
            save_single_path_tcxyz=True,
        ),
    )

    data = scipy.io.loadmat(save_path)
    assert data["W_single_tcxyz"].shape == (3, 1, 3, 3, 3)
    assert data["sol_single_tcxyz"].shape == (3, 1, 3, 3, 3)


def test_kpz_smoke():
    np.random.seed(0)
    x, t, w, sol = kpz_simulator(
        a=0.0,
        b=1.0,
        Nx=4,
        s=0.0,
        t=0.02,
        Nt=4,
        truncation=4,
        sigma=0.1,
        fix_u0=True,
        num=2,
        lam=0.05,
    )

    assert x.shape == (5,)
    assert t.shape == (5,)
    assert w.shape == (2, 5, 5)
    assert sol.shape == (2, 5, 5)


def test_wave_smoke():
    np.random.seed(0)
    x, t, w, sol = wave_simulator(
        a=0.0, b=1.0, Nx=4, s=0.0, t=0.02, Nt=4, truncation=4, fix_u0=True, num=2
    )

    assert x.shape == (5,)
    assert t.shape == (5,)
    assert w.shape == (2, 5, 5)
    assert sol.shape == (2, 5, 5)


def test_kdv_smoke():
    np.random.seed(0)
    x, t, w, sol = kdv_simulator(
        a=0.0,
        b=1.0,
        Nx=8,
        s=0.0,
        t=0.02,
        Nt=4,
        noise_type="cyl",
        sigma=0.1,
        truncation=4,
        fix_u0=True,
        num=2,
    )

    assert x.shape == (9,)
    assert t.shape == (5,)
    assert w.shape == (2, 9, 5)
    assert sol.shape == (2, 8, 5)


def test_kdv_refine_smoke():
    np.random.seed(0)
    x, t, w, sol = kdv_refine_simulator(
        a=0.0,
        b=1.0,
        Nx=8,
        s=0.0,
        t=0.02,
        Nt=4,
        noise_type="cyl",
        sigma=0.1,
        truncation=4,
        fix_u0=True,
        num=2,
    )

    assert x.shape == (9,)
    assert t.shape == (5,)
    assert w.shape == (2, 9, 5)
    assert sol.shape == (2, 8, 5)


def test_navier_stokes_smoke(tmp_path):
    np.random.seed(0)
    cfg = SimpleNamespace(
        s=8,
        alpha=2.0,
        tau=3.0,
        alpha_Q=0.5,
        kappa=1,
        sigma=0.05,
        truncation=8,
        T=0.01,
        delta_t=0.01,
        save_dir=f"{tmp_path}/",
        fix_u0=True,
        bsize=1,
        sub_x=1,
        sub_t=1,
        N=1,
        nu=1e-3,
    )

    folder = ns_simulator(cfg)
    output_path = os.path.join(folder, "NS_small_0.mat")
    data = scipy.io.loadmat(output_path)

    assert os.path.isdir(folder)
    assert os.path.exists(output_path)
    assert data["sol"].shape == (1, 8, 8, 2)
    assert data["forcing"].shape == (1, 8, 8, 2)
    assert data["t"].shape == (1, 2)
