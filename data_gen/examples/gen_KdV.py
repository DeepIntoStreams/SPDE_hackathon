# Portions of this code adapted from https://github.com/crispitagorico/torchspde
# Modified for current implementation by the authors of SPDEBench

import hydra
from omegaconf import DictConfig
import scipy.io
import numpy as np
import os
import os.path as osp
import sys
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(osp.join(current_directory, "..",".."))
from data_gen.src.Noise import Noise
from data_gen.src.general_solver import general_1d_solver


# smooth Q noise as in Example 10.8 of `An Introduction to Computational Stochastic PDEs' by Lord, Powell & Shardlow
def smooth_corr(x, j, a, r):
    def q(j):
        if j == 0:
            return 0
        return (j // 2 + 1) ** (-r)

    return np.sqrt(q(j)) * np.sqrt(2 / a) * np.sin(j * np.pi * x / a)

def simulator(a, b, Nx, s, t, Nt, noise_type, sigma, truncation, fix_u0, num):
    dx, dt = (b-a)/Nx, (t-s)/Nt  # space-time increments
    O_X, O_T = Noise().partition(a,b,dx), Noise().partition(s,t,dt) # space grid O_X and time grid O_T

    u0 = np.array([[np.sin(2*np.pi*x) for x in np.linspace(a, b, Nx + 1)[:-1]] for _ in range(num)])  # initial condition
    if not fix_u0:  # varying initial condition
        X_ = np.linspace(-0.5, 0.5, Nx + 1)
        ic_ = Noise().initial(num, X_, scaling=1, Dirichlet=True)[..., :-1]  # one cycle
        u0 = (ic_ - ic_[:, 0, None]) + u0
        print("u0 is varying!")
    else:
        print("u0 is fixed!")

    # stochastic forcing
    if noise_type == 'cyl':
        r = 4  # Creates r/2 spatially smooth noise
        corr = lambda x, j, a: smooth_corr(x, j, a, r + 1.001)
        W_smooth = Noise().WN_space_time_many(s, t, dt * 0.1, a, b, dx, num, J=truncation, correlation=corr)
        W_smooth = W_smooth[:, ::10, :]
    elif noise_type == 'Q':
        W_smooth = Noise().WN_space_time_many(s, t, dt, a, b, dx, num, J=truncation)
    else:
        print('Invalid noise type!')
        exit(0)

    L_kdv = [0, 0, 1e-3, -0.1, 0]
    mu_kdv = lambda x: 0
    sigma_kdv = lambda x: sigma

    KdV, _, _ = general_1d_solver(L_kdv, u0, W_smooth[:, :, :-1], mu=mu_kdv, sigma=sigma_kdv, Burgers=-6)

    W = W_smooth.transpose(0,2,1)
    soln = KdV.transpose(0,2,1)
    return O_X, O_T, W, soln

@hydra.main(version_base=None, config_path="../configs/", config_name="KdV")
def main(cfg: DictConfig):
    np.random.seed(cfg.seed)

    O_X, O_T, W, soln = simulator(**cfg.sim)

    # Save data
    os.makedirs(cfg.save_dir, exist_ok=True)
    ic_type = 'xi' if cfg.sim.fix_u0 else 'u0_xi'
    filename = f'{cfg.save_name}{cfg.sim.noise_type}_{ic_type}_trc{cfg.sim.truncation}_{cfg.sim.num}.mat'
    scipy.io.savemat(cfg.save_dir + filename, mdict={'X':O_X, 'T':O_T, 'W': W, 'sol': soln})
    print("Saved to", cfg.save_dir + filename)


if __name__ == "__main__":
    main()