import os
import os.path as osp
import sys

import hydra
import numpy as np
import scipy.io
import torch
from omegaconf import DictConfig
from tqdm import tqdm

current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(osp.join(current_directory, "..", ".."))

from data_gen.src.NoiseND import NoiseND
from data_gen.src.SPDEs3D import SPDE3D


def build_initial_conditions(num, grid, fix_u0):
    if fix_u0:
        print("u0 is fixed!")
        return np.zeros((num, len(grid), len(grid), len(grid)), dtype=np.float32)

    noise = NoiseND()
    ic = 0.1 * noise.initial(num, (grid, grid, grid), scaling=1)
    ic = ic - ic[:, :1, :1, :1]
    print("u0 is varying!")
    return ic.astype(np.float32)


def single_path_tcxyz(batch_path, index=0):
    return np.expand_dims(batch_path[index], axis=1)


def build_save_dict(x, y, z, t, w, sol, spde, pre, save_single_path_tcxyz):
    mdict = {
        "X": np.asarray(x, dtype=np.float64),
        "Y": np.asarray(y, dtype=np.float64),
        "Z": np.asarray(z, dtype=np.float64),
        "T": np.asarray(t, dtype=np.float64),
        "W": np.asarray(w, dtype=np.float32),
        "sol": np.asarray(sol, dtype=np.float32),
        "N": np.array(spde.N, dtype=np.int32),
        "dt": np.array(spde.dt, dtype=np.float64),
        "eps": np.array(spde.eps, dtype=np.float64),
        "Cmass": np.array(pre.Cmass, dtype=np.float64),
    }
    if save_single_path_tcxyz:
        mdict["W_single_tcxyz"] = single_path_tcxyz(w).astype(np.float32)
        mdict["sol_single_tcxyz"] = single_path_tcxyz(sol).astype(np.float32)
    return mdict


def simulator(N, dt, steps, num, fix_u0, num_tau, tau_max_multiplier, include_c12, seed=0):
    spde = SPDE3D(
        N=N,
        dt=dt,
        steps=steps,
        seed=seed,
        num_tau=num_tau,
        tau_max_multiplier=tau_max_multiplier,
        include_c12=include_c12,
    )
    pre = spde.precompute()

    grid = spde.spatial_grid()
    time = spde.time_grid()
    initial_conditions = build_initial_conditions(num=num, grid=grid, fix_u0=fix_u0)

    w = np.zeros((num, steps + 1, spde.M, spde.M, spde.M), dtype=np.float32)
    sol = np.zeros_like(w)

    for index in tqdm(range(num)):
        generator = torch.Generator(device=spde.device)
        generator.manual_seed(int(seed) + index)
        phi0 = initial_conditions[index]
        _, trajectory, noise = spde.rollout(phi0=phi0, pre=pre, generator=generator)

        noise_np = noise.detach().cpu().numpy().astype(np.float32)
        sol[index] = trajectory.detach().cpu().numpy().astype(np.float32)
        w[index, 1:] = np.cumsum(noise_np, axis=0, dtype=np.float32)

    return grid, grid, grid, time, w, sol, spde, pre


@hydra.main(version_base=None, config_path="../configs/", config_name="phi43")
def main(cfg: DictConfig):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    x, y, z, t, w, sol, spde, pre = simulator(**cfg.sim, seed=cfg.seed)

    os.makedirs(cfg.save_dir, exist_ok=True)
    ic_type = "xi" if cfg.sim.fix_u0 else "u0_xi"
    filename = f"{cfg.save_name}_{ic_type}_N{cfg.sim.N}_steps{cfg.sim.steps}_{cfg.sim.num}.mat"
    save_path = os.path.join(cfg.save_dir, filename)

    mdict = build_save_dict(
        x=x,
        y=y,
        z=z,
        t=t,
        w=w,
        sol=sol,
        spde=spde,
        pre=pre,
        save_single_path_tcxyz=cfg.save_single_path_tcxyz,
    )
    scipy.io.savemat(save_path, mdict=mdict)

    print("Saved to", save_path)
    print("X shape:", x.shape)
    print("Y shape:", y.shape)
    print("Z shape:", z.shape)
    print("T shape:", t.shape)
    print("W shape:", w.shape)
    print("sol shape:", sol.shape)
    if cfg.save_single_path_tcxyz:
        print("W_single_tcxyz shape:", mdict["W_single_tcxyz"].shape)
        print("sol_single_tcxyz shape:", mdict["sol_single_tcxyz"].shape)


if __name__ == "__main__":
    main()
