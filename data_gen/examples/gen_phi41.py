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
sys.path.append(osp.join(current_directory, "..", ".."))
from data_gen.src.Noise import Noise
from data_gen.src.SPDEs import SPDE


def simulator(a, b, Nx, s, t, Nt, truncation, sigma, fix_u0, num):
    dx, dt = (b-a)/Nx, (t-s)/Nt  # space-time increments
    O_X, O_T = Noise().partition(a,b,dx), Noise().partition(s,t,dt) # space grid O_X and time grid O_T

    mu = lambda x: 3*x-x**3 # drift
    # sigma_fun = lambda x: sigma   # additive diffusive term

    ic = lambda x: x*(1-x) # initial condition (fixed part)
    if not fix_u0: # varying initial condition
        X_ = np.linspace(-0.5,0.5,Nx+1)
        ic_ = Noise().initial(num, X_, scaling = 1) # one cycle
        ic = 0.1*(ic_ - ic_[:,0,None]) + ic(O_X)
        print("u0 is varying!")
    else:
        print("u0 is fixed!")

    W = Noise().WN_space_time_many(s, t, dt, a, b, dx, num, J=truncation) # create realizations of space-time white noise
    Soln_add = SPDE(BC = 'P', IC = ic, mu = mu, sigma = sigma).Parabolic(W, O_T, O_X) # solve parabolic equation

    W = W.transpose(0,2,1)
    Soln_add = Soln_add.transpose(0,2,1)

    return O_X, O_T, W, Soln_add

@hydra.main(version_base=None, config_path="../configs/", config_name="phi41")
def main(cfg: DictConfig):
    np.random.seed(cfg.seed)
    O_X, O_T, W, Soln_add = simulator(**cfg.sim)

    sigma_type = '01' if cfg.sim.sigma == 0.1 else '1'
    ic_type = 'xi' if cfg.sim.fix_u0 else 'u0_xi'
    filename = f'{cfg.save_name}sigma{sigma_type}_{ic_type}_trc{cfg.sim.truncation}_{cfg.sim.num}.mat'

    os.makedirs(cfg.save_dir, exist_ok=True)
    scipy.io.savemat(cfg.save_dir + filename, mdict={'X': O_X, 'T': O_T, 'W': W, 'sol': Soln_add})
    print("Saved to", cfg.save_dir + filename)

    print("X shape: ", O_X.shape)
    print("T shape: ", O_T.shape)
    print("W shape: ", W.shape)
    print("sol shape: ", Soln_add.shape)

if __name__ == "__main__":
    main()