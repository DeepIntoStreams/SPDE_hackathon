import argparse
import os
import os.path as osp
import sys

import numpy as np
import scipy.io


CURRENT_DIR = osp.dirname(osp.abspath(__file__))
SPDE_ROOT = osp.abspath(osp.join(CURRENT_DIR, "..", ".."))

if SPDE_ROOT not in sys.path:
    sys.path.insert(0, SPDE_ROOT)

from data_gen.src.NoiseND import NoiseND  # noqa: E402
from data_gen.src.SPDEs import SPDE  # noqa: E402


def simulator(a, b, nx, s, t, nt, truncation, sigma, fix_u0, num, seed):
    np.random.seed(seed)

    noise = NoiseND()
    dx, dt = (b - a) / nx, (t - s) / nt
    o_x = noise.partition_axis(a, b, dx)
    o_t = noise.partition_axis(s, t, dt)

    mu = lambda x: 3 * x - x**3
    ic_base = lambda x: x * (1 - x)

    if fix_u0:
        ic = ic_base
        print("u0 is fixed!")
    else:
        x_grid = np.linspace(-0.5, 0.5, nx + 1)
        ic_noise = noise.initial(num, (x_grid,), scaling=1)[:, :]
        ic = 0.1 * (ic_noise - ic_noise[:, 0, None]) + ic_base(o_x)
        print("u0 is varying!")

    w = noise.WN_space_time_many(
        s,
        t,
        dt,
        bounds=((a, b),),
        steps=(dx,),
        num=num,
        truncation=(truncation + 1,),
    )
    sol = SPDE(BC="P", IC=ic, mu=mu, sigma=sigma).Parabolic(w, o_t, o_x)

    return o_x, o_t, w.transpose(0, 2, 1), sol.transpose(0, 2, 1)


def sigma_label(sigma):
    if abs(sigma - 0.1) < 1e-12:
        return "01"
    if float(sigma).is_integer():
        return str(int(sigma))
    return str(sigma).replace(".", "p")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Phi41 sigma=1 datasets for multiple noise truncations J.")
    parser.add_argument("--j-values", type=int, nargs="+", default=[2, 8, 32, 64, 128, 256])
    parser.add_argument("--num", type=int, default=1200)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--nt", type=int, default=50)
    parser.add_argument("--a", type=float, default=0.0)
    parser.add_argument("--b", type=float, default=1.0)
    parser.add_argument("--s", type=float, default=0.0)
    parser.add_argument("--t", type=float, default=0.05)
    parser.add_argument("--fix-u0", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-dir", type=str, default=osp.join(CURRENT_DIR, "data"))
    parser.add_argument("--save-prefix", type=str, default="Phi41")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    ic_type = "xi" if args.fix_u0 else "u0_xi"
    sig_type = sigma_label(args.sigma)

    for j in args.j_values:
        filename = f"{args.save_prefix}_sigma{sig_type}_{ic_type}_trc{j}_{args.num}.mat"
        save_path = osp.join(args.save_dir, filename)

        if osp.exists(save_path) and not args.overwrite:
            print(f"Skipping existing file: {save_path}")
            continue

        print("=" * 80)
        print(f"Generating Phi41 sigma={args.sigma}, J={j}, num={args.num}, seed={args.seed}")

        o_x, o_t, w, sol = simulator(
            a=args.a,
            b=args.b,
            nx=args.nx,
            s=args.s,
            t=args.t,
            nt=args.nt,
            truncation=j,
            sigma=args.sigma,
            fix_u0=args.fix_u0,
            num=args.num,
            seed=args.seed,
        )

        scipy.io.savemat(
            save_path,
            mdict={
                "X": o_x,
                "T": o_t,
                "W": w,
                "sol": sol,
                "sigma": args.sigma,
                "truncation": j,
                "seed": args.seed,
            },
        )

        print(f"Saved to {save_path}")
        print("X shape:", o_x.shape)
        print("T shape:", o_t.shape)
        print("W shape:", w.shape)
        print("sol shape:", sol.shape)


if __name__ == "__main__":
    main()
