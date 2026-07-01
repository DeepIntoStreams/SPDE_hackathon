# Portions of this code adapted from https://github.com/crispitagorico/torchspde
# Modified for current implementation by the authors of SPDEBench

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
from data_gen.src.SPDEs import SPDE

def simulator(a, b, Nx, s, t, Nt, truncation, sigma, fix_u0, num, lam, basis="sincos", boundary="Periodic"):
    # enforce basis/boundary combination inside simulator
    valid_combinations = {
        "sincos": "Periodic",
    }
    if basis not in valid_combinations or valid_combinations[basis] != boundary:
        raise ValueError(f"Invalid combination of basis '{basis}' and boundary '{boundary}'.")

    noise = NoiseND(basis=basis, covariance="cylindrical")
    dx, dt = (b-a)/Nx, (t-s)/Nt  # space-time increments
    O_X = noise.partition_axis(a, b, dx)
    O_T = noise.partition_axis(s, t, dt)

    mu = lambda x: 3*x-x**3 # drift

    ic = lambda x: x*(1-x) # initial condition (fixed part)
    if not fix_u0: # varying initial condition
        X_ = np.linspace(-0.5,0.5,Nx+1)
        ic_ = noise.initial(num, (X_,))[:, :]
        ic = 0.1*(ic_ - ic_[:,0,None]) + ic(O_X)
        print("u0 is varying!")
    else:
        print("u0 is fixed!")

    W = noise.WN_space_time_KPZ(
        s,
        t,
        dt,
        bounds=((a, b),),
        steps=(dx,),
        truncation=(truncation + 1,),
        num=num,
    )
    Soln_add = SPDE(BC=boundary, IC=ic, mu=mu, sigma=sigma).Parabolic_reno(W, O_T, O_X, lam, truncation) # solve parabolic equation

    W = W.transpose(0,2,1)
    Soln_add = Soln_add.transpose(0,2,1)

    return O_X, O_T, W, Soln_add

@hydra.main(version_base=None, config_path="../configs/", config_name="KPZ")
def main(cfg: DictConfig):

    truncations = OmegaConf.to_container(cfg.sim.truncation) if OmegaConf.is_list(cfg.sim.truncation) else [cfg.sim.truncation]
    sigmas = OmegaConf.to_container(cfg.sim.sigma) if OmegaConf.is_list(cfg.sim.sigma) else [cfg.sim.sigma]
    fix_u0_options = OmegaConf.to_container(cfg.sim.fix_u0) if OmegaConf.is_list(cfg.sim.fix_u0) else [cfg.sim.fix_u0]
    lams = OmegaConf.to_container(cfg.sim.lam) if OmegaConf.is_list(cfg.sim.lam) else [cfg.sim.lam]

    # Generate all combinations of the parameters
    combinations = product(truncations, sigmas, fix_u0_options, lams)

    for truncation, sigma, fix_u0, lam in combinations:
        print(f"Processing combination: truncation={truncation}, sigma={sigma}, fix_u0={fix_u0}, lam={lam}")

        try:
            np.random.seed(cfg.seed)
            sim_params = dict(cfg.sim)
            sim_params["truncation"] = truncation
            sim_params["sigma"] = sigma
            sim_params["fix_u0"] = fix_u0
            sim_params["lam"] = lam
            # explicit basis/boundary for KPZ
            basis = cfg.sim.get("basis", "sincos")
            boundary = cfg.sim.get("boundary", "Periodic")
            # remove potential duplicate keys before passing
            sim_params.pop("basis", None)
            sim_params.pop("boundary", None)
            O_X, O_T, W, Soln_add = simulator(**sim_params, basis=basis, boundary=boundary)
        except Exception as e:
            print(f"Skipping combination: {e}")
            continue

        sigma_type = '01' if sigma == 0.1 else '1'
        ic_type = 'xi' if fix_u0 else 'u0_xi'
        lam_type = '005' if lam == 0.05 else '05'
        filename = f'{cfg.save_name}sigma{sigma_type}_{ic_type}_lam{lam_type}_trc{truncation}_{cfg.sim.num}.mat'

        save_path = os.path.join(cfg.save_dir, filename)
        os.makedirs(cfg.save_dir, exist_ok=True)
        mdict = {
            'X': np.array(O_X, dtype=np.float64),
            'T': np.array(O_T, dtype=np.float64),
            'W': np.array(W, dtype=np.float64),
            'sol': np.array(Soln_add, dtype=np.float64),
        }
        scipy.io.savemat(save_path, mdict=mdict)
        print("Saved to", cfg.save_dir + filename)

        print("X shape: ", O_X.shape)
        print("T shape: ", O_T.shape)
        print("W shape: ", W.shape)
        print("sol shape: ", Soln_add.shape)

if __name__ == "__main__":
    main()
