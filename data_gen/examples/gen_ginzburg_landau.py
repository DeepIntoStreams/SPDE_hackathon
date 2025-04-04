import os
import hydra
from omegaconf import DictConfig
import scipy.io
import numpy as np
from data_gen.src.Noise import Noise, partition
from data_gen.src.SPDEs import SPDE

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def simulator(a, b, Nx, s, t, Nt, num):
    dx, dt = (b-a)/Nx, (t-s)/Nt  # space-time increments
    O_X, O_T = partition(a,b,dx), partition(s,t,dt) # space grid O_X and time grid O_T

    mu = lambda x: 3*x-x**3 # drift
    sigma = lambda x: 1 # additive diffusive term
    ic = lambda x: x*(1-x) # initial condition (fixed)
    # Another kind of initial condition (varying)
    # X_ = np.linspace(-0.5,0.5,129)
    # ic_ = Noise().initial(n, X_, scaling = 1) # one cycle
    # ic = 0.1*(ic_-ic_[:,0,None]) + ic(O_X)

    W = Noise().WN_space_time_many(s, t, dt, a, b, dx, num) # create realizations of space-time white noise
    Soln_add = SPDE(BC = 'P', IC = ic, mu = mu, sigma = sigma).Parabolic(0.1*W, O_T, O_X) # solve parabolic equation
    W = W.transpose(0,2,1)
    soln = Soln_add.transpose(0,2,1)
    return O_X, O_T, W, soln

@hydra.main(version_base=None, config_path="../configs/", config_name="ginzburg_landau")
def main(cfg: DictConfig):
    np.random.seed(cfg.seed)
    O_X, O_T, W, soln = simulator(**cfg.sim)
    os.makedirs(cfg.save_dir, exist_ok=True)
    scipy.io.savemat(cfg.save_dir + 'Phi41+_xi_{}.mat'.format(cfg.sim.num), mdict={'X':O_X, 'T':O_T, 'W': W, 'sol': soln})

if __name__ == "__main__":
    main()