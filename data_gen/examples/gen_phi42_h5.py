import hydra
import h5py
import numpy as np
import os
import os.path as osp
import sys
from omegaconf import DictConfig, OmegaConf
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(osp.join(current_directory, "..", ".."))
from data_gen.src.NoiseND import NoiseND
from data_gen.src.SPDEs2D import SPDE2D


def save_h5(path, arrays, dtypes=None, compression="gzip", compression_opts=4, chunks=True):
    os.makedirs(osp.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as handle:
        for key, value in arrays.items():
            if dtypes and key in dtypes:
                value = np.asarray(value, dtype=dtypes[key])
            else:
                value = np.asarray(value)
            kwargs = {}
            if compression is not None:
                kwargs["compression"] = compression
                kwargs["compression_opts"] = compression_opts
            if chunks:
                kwargs["chunks"] = True
            handle.create_dataset(key, data=value, **kwargs)


def create_or_resize_dataset(handle, name, shape, maxshape, dtype, chunks=True, compression="gzip", compression_opts=4):
    if name in handle:
        ds = handle[name]
        if ds.shape != shape:
            ds.resize(shape)
        return ds
    kwargs = {"dtype": dtype, "maxshape": maxshape}
    if compression is not None:
        kwargs["compression"] = compression
        kwargs["compression_opts"] = compression_opts
    if chunks:
        kwargs["chunks"] = True
    return handle.create_dataset(name, shape, **kwargs)


def solver(a, b, Nx, c, d, Ny, s, t, Nt, num, eps, sigma, fix_u0):
    basis = "sincos"
    boundary = "Periodic"
    noise = NoiseND(basis=basis)

    dx, dy, dt = (b-a)/Nx, (d-c)/Ny, (t - s) / Nt
    ic = lambda x, y: np.sin(2 * np.pi * (x + y)) + np.cos(2 * np.pi * (x + y))
    mu = lambda x: 3*x-x**3

    O_X, O_Y = noise.partition_nd(((a, b), (c, d)), (dx, dy))
    O_T = noise.partition_axis(s, t, dt)
    W = noise.WN_space_time(
        s,
        t,
        dt,
        bounds=((a, b), (c, d)),
        steps=(dx, dy),
        truncation=(eps, eps),
        num=num,
    )

    if not fix_u0:
        grid_X, grid_Y = np.meshgrid(O_X, O_Y)
        ic_ = 0.1 * noise.initial(num, (O_X, O_Y))
        ic = 0.1 * (ic_ - ic_[:, 0, None, 0, None]) + ic(grid_X, grid_Y)

    soln_reno = SPDE2D(BC=boundary, IC=ic, mu=mu, sigma=sigma).Renormalization(W, O_T, O_X, O_Y, trcation=eps)
    return O_X, O_Y, O_T, W.astype("float32"), soln_reno.astype("float32")


def append_batch_to_h5(file_path, batch_index, batch_size, total_size, O_X, O_Y, O_T, W, soln_reno):
    mode = "w" if batch_index == 0 else "a"
    with h5py.File(file_path, mode) as handle:
        if batch_index == 0:
            handle.create_dataset("X", data=O_X, dtype=O_X.dtype)
            handle.create_dataset("Y", data=O_Y, dtype=O_Y.dtype)
            handle.create_dataset("T", data=O_T, dtype=O_T.dtype)
            shape = (total_size, ) + W.shape[1:]
            maxshape = (None, ) + W.shape[1:]
            handle.create_dataset("W", shape, maxshape=maxshape, dtype=W.dtype, chunks=(1,) + W.shape[1:], compression="gzip", compression_opts=4)
            handle.create_dataset("sol_reno", shape, maxshape=maxshape, dtype=soln_reno.dtype, chunks=(1,) + soln_reno.shape[1:], compression="gzip", compression_opts=4)
            handle.attrs["generated_by"] = "gen_phi42_h5.py"
            handle.attrs["batch_size"] = batch_size
            handle.attrs["total_size"] = total_size
        start = batch_index * batch_size
        end = start + W.shape[0]
        handle["W"][start:end, ...] = W
        handle["sol_reno"][start:end, ...] = soln_reno


@hydra.main(version_base=None, config_path="../configs/", config_name="phi42")
def main(cfg: DictConfig):
    np.random.seed(cfg.seed)
    batch_size = int(cfg.batch_size) if hasattr(cfg, "batch_size") else 20
    total_size = int(cfg.sim.num)
    if total_size % batch_size != 0:
        raise ValueError("cfg.sim.num must be divisible by batch_size for consistent batching.")
    num_batches = total_size // batch_size

    eps_list = OmegaConf.to_container(cfg.sim.eps) if OmegaConf.is_list(cfg.sim.eps) else [cfg.sim.eps]
    sigma_list = OmegaConf.to_container(cfg.sim.sigma) if OmegaConf.is_list(cfg.sim.sigma) else [cfg.sim.sigma]
    fix_u0_options = OmegaConf.to_container(cfg.sim.fix_u0) if OmegaConf.is_list(cfg.sim.fix_u0) else [cfg.sim.fix_u0]

    for eps in eps_list:
        for sigma in sigma_list:
            for fix_u0 in fix_u0_options:
                ic_type = "xi" if fix_u0 else "xi_u0"
                output_path = osp.join(cfg.save_dir, f"{cfg.save_name}_{ic_type}_eps{eps}_sigma{sigma}_seed{cfg.seed}_{total_size}.h5")
                os.makedirs(cfg.save_dir, exist_ok=True)

                print(f"Generating data for eps={eps}, sigma={sigma}, fix_u0={fix_u0}, total={total_size}, batch_size={batch_size}")
                for batch_index in range(num_batches):
                    np.random.seed(cfg.seed + batch_index)
                    current_size = batch_size
                    if batch_index == num_batches - 1:
                        current_size = total_size - batch_index * batch_size
                    O_X, O_Y, O_T, W, soln_reno = solver(
                        cfg.sim.a,
                        cfg.sim.b,
                        cfg.sim.Nx,
                        cfg.sim.c,
                        cfg.sim.d,
                        cfg.sim.Ny,
                        cfg.sim.s,
                        cfg.sim.t,
                        cfg.sim.Nt,
                        current_size,
                        eps,
                        sigma,
                        fix_u0,
                    )
                    append_batch_to_h5(output_path, batch_index, batch_size, total_size, O_X, O_Y, O_T, W, soln_reno)
                    print(f"Saved batch {batch_index + 1}/{num_batches} to {output_path}")

                print(f"Finished generating {output_path}")


if __name__ == "__main__":
    main()
