import os
import os.path as osp
import sys

import h5py
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

current_directory = os.path.dirname(os.path.abspath(__file__))
new_data_root = osp.abspath(osp.join(current_directory, ".."))
if new_data_root not in sys.path:
    sys.path.insert(0, new_data_root)

from src.NoiseND import NoiseND
from src.SPDEs3D import SPDE3D


def compression_arg(name):
    if name is None:
        return None
    if str(name).lower() in {"none", "null", "false"}:
        return None
    return str(name)


def build_initial_conditions(num, grid, fix_u0):
    shape = (num, len(grid), len(grid), len(grid))
    if fix_u0:
        print("u0 is fixed!")
        return np.zeros(shape, dtype=np.float32)

    noise = NoiseND()
    ic = 0.1 * noise.initial(num, (grid, grid, grid), scaling=1)
    ic = ic - ic[:, :1, :1, :1]
    print("u0 is varying!")
    return ic.astype(np.float32)


def create_spde(N, dt, steps, num_tau, tau_max_multiplier, include_c12, seed=0):
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
    return spde, pre


def default_filename(save_name, fix_u0, N, steps, num):
    ic_type = "xi" if fix_u0 else "u0_xi"
    return f"{save_name}_{ic_type}_N{N}_steps{steps}_{num}.h5"


def create_output_file(path, spde, pre, num, compression, gzip_level, overwrite):
    if osp.exists(path) and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Set +overwrite=true to replace it.")

    os.makedirs(osp.dirname(path), exist_ok=True)
    compression_opts = gzip_level if compression == "gzip" else None

    grid = spde.spatial_grid()
    time = spde.time_grid()
    sample_shape = (spde.steps + 1, spde.M, spde.M, spde.M)

    h5 = h5py.File(path, "w")
    h5.create_dataset("X", data=np.asarray(grid, dtype=np.float64))
    h5.create_dataset("Y", data=np.asarray(grid, dtype=np.float64))
    h5.create_dataset("Z", data=np.asarray(grid, dtype=np.float64))
    h5.create_dataset("T", data=np.asarray(time, dtype=np.float64))
    h5.create_dataset("N", data=np.array(spde.N, dtype=np.int32))
    h5.create_dataset("dt", data=np.array(spde.dt, dtype=np.float64))
    h5.create_dataset("eps", data=np.array(spde.eps, dtype=np.float64))
    h5.create_dataset("Cmass", data=np.array(pre.Cmass, dtype=np.float64))

    h5.create_dataset(
        "W",
        shape=(num,) + sample_shape,
        dtype=np.float32,
        chunks=(1,) + sample_shape,
        compression=compression,
        compression_opts=compression_opts,
    )
    h5.create_dataset(
        "sol",
        shape=(num,) + sample_shape,
        dtype=np.float32,
        chunks=(1,) + sample_shape,
        compression=compression,
        compression_opts=compression_opts,
    )

    h5.attrs["total_samples"] = int(num)
    h5.attrs["N"] = int(spde.N)
    h5.attrs["steps"] = int(spde.steps)
    h5.attrs["dt"] = float(spde.dt)
    h5.attrs["eps"] = float(spde.eps)
    h5.attrs["Cmass"] = float(pre.Cmass)
    h5.attrs["compression"] = "none" if compression is None else str(compression)
    return h5


@hydra.main(version_base=None, config_path="../configs/", config_name="phi43")
def main(cfg: DictConfig):
    np.random.seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))

    num = int(cfg.get("total_num", cfg.sim.num))
    if num <= 0:
        raise ValueError("total_num/sim.num must be positive.")

    compression = compression_arg(cfg.get("compression", "lzf"))
    gzip_level = int(cfg.get("gzip_level", 4))
    overwrite = bool(cfg.get("overwrite", False))

    spde, pre = create_spde(
        N=cfg.sim.N,
        dt=cfg.sim.dt,
        steps=cfg.sim.steps,
        num_tau=cfg.sim.num_tau,
        tau_max_multiplier=cfg.sim.tau_max_multiplier,
        include_c12=cfg.sim.include_c12,
        seed=cfg.seed,
    )

    filename = cfg.get("output_name", None)
    if filename is None:
        filename = default_filename(
            save_name=cfg.save_name,
            fix_u0=cfg.sim.fix_u0,
            N=cfg.sim.N,
            steps=cfg.sim.steps,
            num=num,
        )
    output_path = osp.join(cfg.save_dir, filename)

    grid = spde.spatial_grid()
    initial_conditions = build_initial_conditions(num=num, grid=grid, fix_u0=cfg.sim.fix_u0)

    print("Output path:", output_path)
    print("Total samples:", num)
    print("W/sol shape:", (num, spde.steps + 1, spde.M, spde.M, spde.M))
    print("Compression:", "none" if compression is None else compression)

    with create_output_file(
        path=output_path,
        spde=spde,
        pre=pre,
        num=num,
        compression=compression,
        gzip_level=gzip_level,
        overwrite=overwrite,
    ) as h5:
        for index in tqdm(range(num)):
            generator = torch.Generator(device=spde.device)
            generator.manual_seed(int(cfg.seed) + index)
            phi0 = initial_conditions[index]
            _, trajectory, noise = spde.rollout(phi0=phi0, pre=pre, generator=generator)

            noise_np = noise.detach().cpu().numpy().astype(np.float32)
            w_path = np.zeros((spde.steps + 1, spde.M, spde.M, spde.M), dtype=np.float32)
            w_path[1:] = np.cumsum(noise_np, axis=0, dtype=np.float32)
            sol_path = trajectory.detach().cpu().numpy().astype(np.float32)

            h5["W"][index] = w_path
            h5["sol"][index] = sol_path
            if (index + 1) % int(cfg.get("flush_every", 10)) == 0:
                h5.flush()

        h5.flush()

    print("Saved Phi43 HDF5 to:", output_path)


if __name__ == "__main__":
    main()
