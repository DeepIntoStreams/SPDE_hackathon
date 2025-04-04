import os
import hydra
from omegaconf import DictConfig
import scipy.io
import numpy as np
from data_gen.src.Noise import Noise, partition
from data_gen.src.SPDEs import SPDE

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def simulator(cfg):
    n = cfg.sim.num
    a, b, s, t = cfg.sim.a, cfg.sim.b, cfg.sim.s, cfg.sim.t
    Nx, Nt = cfg.sim.Nx, cfg.sim.Nt
    dx, dt = (b-a)/Nx, (t-s)/Nt  # space-time increments
    O_X, O_T = partition(a,b,dx), partition(s,t,dt) # space grid O_X and time grid O_T

    ic = lambda x: np.sin(2 * np.pi * x)  # initial condition
    ic_t = lambda x: x * (1 - x)  # initial speed
    mu = lambda x: np.cos(np.pi * x) + x ** 2  # drift
    sigma = lambda x: x  # diffusion

    W = Noise().WN_space_time_many(s, t, dt, a, b, dx, n)  # Create realizations of space time white noise
    # solve wave equation
    Wave_soln = SPDE(Type='W', BC='P', T=O_T, X=O_X, IC=ic, IC_t=ic_t, mu=mu, sigma=sigma).Wave(W)

    W = W.transpose(0,2,1)
    soln = Wave_soln.transpose(0,2,1)
    return O_X, O_T, W, soln

@hydra.main(version_base=None, config_path="../configs/", config_name="wave")
def main(cfg: DictConfig):
    np.random.seed(cfg.seed)
    n = cfg.sim.num
    O_X, O_T, W, soln = simulator(cfg)
    os.makedirs(cfg.save_dir, exist_ok=True)
    scipy.io.savemat(cfg.save_dir + 'wave_xi_{}.mat'.format(n), mdict={'X':O_X, 'T':O_T, 'W': W, 'sol': soln})

if __name__ == "__main__":
    main()