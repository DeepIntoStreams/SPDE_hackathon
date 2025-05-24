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
from data_gen.src.SPDEs import SPDE


def simulator(a, b, Nx, s, t, Nt, truncation, fix_u0, num):
    dx, dt = (b-a)/Nx, (t-s)/Nt  # space-time increments
    O_X, O_T = Noise().partition(a,b,dx), Noise().partition(s,t,dt) # space grid O_X and time grid O_T

    ic = lambda x: np.sin(2 * np.pi * x)  # initial condition (fixed part)
    if not fix_u0:  # varying initial condition
        X_ = np.linspace(-0.5,0.5,Nx+1)
        ic_ = Noise().initial(num, X_, scaling = 1)
        ic = (ic_-ic_[:,0,None]) + ic(O_X)
        print("u0 is varying!")
    else:
        print("u0 is fixed!")

    ic_t = lambda x: x * (1 - x)  # initial speed
    mu = lambda x: np.cos(np.pi * x) + x ** 2  # drift
    sigma = lambda x: x  # diffusion

    W = Noise().WN_space_time_many(s, t, dt, a, b, dx, num, J=truncation)  # Create realizations of space time white noise
    Wave_soln = SPDE(Type='W', BC='P', T=O_T, X=O_X, IC=ic, IC_t=ic_t, mu=mu, sigma=sigma).Wave(W)  # solve wave equation

    W = W.transpose(0,2,1)
    soln = Wave_soln.transpose(0,2,1)
    return O_X, O_T, W, soln

@hydra.main(version_base=None, config_path="../configs/", config_name="wave")
def main(cfg: DictConfig):
    np.random.seed(cfg.seed)
    O_X, O_T, W, soln = simulator(**cfg.sim)
    os.makedirs(cfg.save_dir, exist_ok=True)
    ic_type = 'xi' if cfg.sim.fix_u0 else 'u0_xi'
    filename = f'{cfg.save_name}_{ic_type}_trc{cfg.sim.truncation}_{cfg.sim.num}.mat'
    scipy.io.savemat(cfg.save_dir + filename, mdict={'X':O_X, 'T':O_T[::5], 'W': W[:,:,::5], 'sol': soln[:,:,::5]})
    print("Saved to", cfg.save_dir + filename)

if __name__ == "__main__":
    main()