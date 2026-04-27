import hydra
from omegaconf import DictConfig, OmegaConf
import scipy.io
import numpy as np
import os
import os.path as osp
import sys
from itertools import product
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(osp.join(current_directory, "..", ".."))
from data_gen.src.NoiseND import NoiseND
from data_gen.src.SPDEs2D import SPDE2D

def solver(a, b, Nx, c, d, Ny, s, t, Nt, num, eps, sigma, fix_u0):
    basis = "sincos"
    boundary = "Periodic"
    noise = NoiseND(basis=basis)

    dx, dy, dt = (b-a)/Nx, (d-c)/Ny, (t - s) / Nt  # space-time increments

    ic = lambda x, y: np.sin(2 * np.pi * (x + y)) + np.cos(2 * np.pi * (x + y)) # initial condition (fixed)

    mu = lambda x: 3*x-x**3 # drift
    # sigma_fun = lambda x: sigma # additive diffusive term

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

    if not fix_u0: # varying initial condition
        grid_X, grid_Y = np.meshgrid(O_X, O_Y)
        ic_ = 0.1 * noise.initial(num, (O_X, O_Y))
        ic = 0.1*(ic_-ic_[:,0,None,0,None]) + ic(grid_X, grid_Y)
        print("u0 is varying!")
    else:
        print("u0 is fixed!")

    Soln_reno = SPDE2D(BC = boundary, IC = ic, mu = mu, sigma = sigma).Renormalization(W, O_T, O_X, O_Y, trcation=eps) # generate through explicit scheme without renormalization
    Soln_expl = SPDE2D(BC = boundary, IC = ic, mu = mu, sigma = sigma).Parabolic(W, O_T, O_X, O_Y) # generate through explicit scheme with renormalization

    return O_X, O_Y, O_T, W, eps, Soln_reno, Soln_expl

@hydra.main(version_base=None, config_path="../configs/", config_name="phi42")
def main(cfg: DictConfig):              
    eps_list = OmegaConf.to_container(cfg.sim.eps) if OmegaConf.is_list(cfg.sim.eps) else [cfg.sim.eps]
    sigma_list = OmegaConf.to_container(cfg.sim.sigma) if OmegaConf.is_list(cfg.sim.sigma) else [cfg.sim.sigma]
    fix_u0_options = OmegaConf.to_container(cfg.sim.fix_u0) if OmegaConf.is_list(cfg.sim.fix_u0) else [cfg.sim.fix_u0]

    combinations = product(eps_list, sigma_list, fix_u0_options)

    for eps, sigma, fix_u0 in combinations:
        print(f"Processing combination: eps={eps}, sigma={sigma}, fix_u0={fix_u0}")

        try:
            np.random.seed(cfg.seed)
            sim_params = dict(cfg.sim)
            sim_params["eps"] = eps
            sim_params["sigma"] = sigma
            sim_params["fix_u0"] = fix_u0
            O_X, O_Y, O_T, W, eps, soln_reno, soln_expl = solver(**sim_params)

            os.makedirs(cfg.save_dir, exist_ok=True)
            ic_type = 'xi' if fix_u0 else 'xi_u0'

            reno_filename = f'{cfg.save_name}_reno_{ic_type}_eps{eps}_{cfg.sim.num}.mat'
            scipy.io.savemat(cfg.save_dir + reno_filename, mdict={'X':O_X, 'Y':O_Y, 'T':O_T, 'W': W, 'eps':eps, 'sol': soln_reno})
            print("Saved to", cfg.save_dir + reno_filename)

            expl_filename = f'{cfg.save_name}_expl_{ic_type}_eps{eps}_{cfg.sim.num}.mat'
            scipy.io.savemat(cfg.save_dir + expl_filename, mdict={'X':O_X, 'Y':O_Y, 'T':O_T, 'W': W, 'eps':eps, 'sol': soln_expl})
            print("Saved to", cfg.save_dir + expl_filename)
        except Exception as e:
            print(f"Skipping combination: {e}")
            continue

if __name__ == "__main__":
    main()
