import argparse
import os
import os.path as osp
import sys
from dataclasses import replace

import h5py
import numpy as np
import torch
from tqdm import tqdm


CURRENT_DIR = osp.dirname(osp.abspath(__file__))
SPDE_ROOT = osp.abspath(osp.join(CURRENT_DIR, "..", ".."))
NEW_DATA_ROOT = osp.join(SPDE_ROOT, "new_data_gen_phi43")
if NEW_DATA_ROOT not in sys.path:
    sys.path.insert(0, NEW_DATA_ROOT)

from src.NoiseND import NoiseND  # noqa: E402
from src.SPDEs3D import SPDE3D  # noqa: E402


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


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


def create_output_file(path, spde, pre, num, compression, gzip_level, overwrite, renorm_mode):
    if osp.exists(path) and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Use --overwrite to replace it.")

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
    h5.create_dataset("C0", data=np.array(pre.C0, dtype=np.float64))
    h5.create_dataset("C11", data=np.array(pre.C11, dtype=np.float64))
    h5.create_dataset("C12", data=np.array(pre.C12, dtype=np.float64))
    h5.create_dataset("C1", data=np.array(pre.C1, dtype=np.float64))
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
    h5.attrs["C0"] = float(pre.C0)
    h5.attrs["C11"] = float(pre.C11)
    h5.attrs["C12"] = float(pre.C12)
    h5.attrs["C1"] = float(pre.C1)
    h5.attrs["Cmass"] = float(pre.Cmass)
    h5.attrs["renorm_mode"] = renorm_mode
    h5.attrs["compression"] = "none" if compression is None else str(compression)
    return h5


def default_output_names(save_name, fix_u0, n, steps, num):
    ic_type = "xi" if fix_u0 else "u0_xi"
    stem = f"{save_name}_{ic_type}_N{n}_steps{steps}_{num}"
    return f"{stem}_full_renorm.h5", f"{stem}_no_renorm.h5"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate paired Phi43 visualization data with full renormalization and Cmass=0."
    )
    parser.add_argument("--N", type=int, default=8, help="Noise/Fourier cutoff.")
    parser.add_argument("--dt", type=float, default=1.384e-4)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--fix-u0", type=str_to_bool, default=True)
    parser.add_argument("--num-tau", type=int, default=48)
    parser.add_argument("--tau-max-multiplier", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-num", type=int, default=1)
    parser.add_argument("--save-dir", type=str, default="/workspace/results/Phi43_with_without_renorm/")
    parser.add_argument("--save-name", type=str, default="Phi43")
    parser.add_argument("--full-output-name", type=str, default=None)
    parser.add_argument("--none-output-name", type=str, default=None)
    parser.add_argument("--compression", type=str, default="lzf")
    parser.add_argument("--gzip-level", type=int, default=4)
    parser.add_argument("--flush-every", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.total_num <= 0:
        raise ValueError("--total-num must be positive.")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    compression = compression_arg(args.compression)

    spde = SPDE3D(
        N=args.N,
        dt=args.dt,
        steps=args.steps,
        seed=args.seed,
        num_tau=args.num_tau,
        tau_max_multiplier=args.tau_max_multiplier,
        include_c12=True,
    )
    pre_full = spde.precompute()
    pre_none = replace(pre_full, Cmass=0.0)

    full_name, none_name = default_output_names(args.save_name, args.fix_u0, args.N, args.steps, args.total_num)
    if args.full_output_name is not None:
        full_name = args.full_output_name
    if args.none_output_name is not None:
        none_name = args.none_output_name

    full_path = osp.join(args.save_dir, full_name)
    none_path = osp.join(args.save_dir, none_name)

    grid = spde.spatial_grid()
    initial_conditions = build_initial_conditions(
        num=args.total_num,
        grid=grid,
        fix_u0=args.fix_u0,
    )

    print("Full renormalization output:", full_path)
    print("No renormalization output:", none_path)
    print("Total samples:", args.total_num)
    print("W/sol shape:", (args.total_num, spde.steps + 1, spde.M, spde.M, spde.M))
    print("Compression:", "none" if compression is None else compression)
    print("Full Cmass:", pre_full.Cmass)
    print("No-renorm Cmass:", pre_none.Cmass)

    with create_output_file(
        path=full_path,
        spde=spde,
        pre=pre_full,
        num=args.total_num,
        compression=compression,
        gzip_level=args.gzip_level,
        overwrite=args.overwrite,
        renorm_mode="full",
    ) as h5_full, create_output_file(
        path=none_path,
        spde=spde,
        pre=pre_none,
        num=args.total_num,
        compression=compression,
        gzip_level=args.gzip_level,
        overwrite=args.overwrite,
        renorm_mode="none",
    ) as h5_none:
        for index in tqdm(range(args.total_num)):
            generator = torch.Generator(device=spde.device)
            generator.manual_seed(args.seed + index)
            phi0 = initial_conditions[index]

            _, trajectory_full, noise = spde.rollout(phi0=phi0, pre=pre_full, generator=generator)
            _, trajectory_none, _ = spde.rollout(phi0=phi0, pre=pre_none, noise=noise)

            noise_np = noise.detach().cpu().numpy().astype(np.float32)
            w_path = np.zeros((spde.steps + 1, spde.M, spde.M, spde.M), dtype=np.float32)
            w_path[1:] = np.cumsum(noise_np, axis=0, dtype=np.float32)

            h5_full["W"][index] = w_path
            h5_full["sol"][index] = trajectory_full.detach().cpu().numpy().astype(np.float32)
            h5_none["W"][index] = w_path
            h5_none["sol"][index] = trajectory_none.detach().cpu().numpy().astype(np.float32)

            if (index + 1) % args.flush_every == 0:
                h5_full.flush()
                h5_none.flush()

        h5_full.flush()
        h5_none.flush()

    print("Saved paired Phi43 visualization HDF5 files.")


if __name__ == "__main__":
    main()
