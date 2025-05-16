import os
import hydra
from omegaconf import DictConfig
import scipy.io
import numpy as np
from data_gen.src.Noise import Noise2D
from data_gen.src.SPDEs2D import SPDE2D

def solver(a, b, Nx, c, d, Ny, s, t, Nt, num, eps):

    dx, dy, dt = (b-a)/Nx, (d-c)/Ny, (t - s) / Nt  # space-time increments

    ic = lambda x, y: np.sin(2 * np.pi * (x + y)) + np.cos(2 * np.pi * (x + y)) # initial condition (fixed)

    mu = lambda x: 3*x-x**3 # drift
    sigma = lambda x: 1 # additive diffusive term

    O_X, O_Y = Noise2D().partition_2d(a,b,dx,c,d,dy) # space grid O_X, O_Y
    O_T = Noise2D().partition(s,t,dt) # time grid O_T
    W = Noise2D().WN_space_time_2d_many(s, t, dt, a, b, dx, c, d, dy, num, eps, eps) # create realizations of space-time white noise

    # Another kind of initial condition (varying)
    # grid_X, grid_Y = np.meshgrid(O_X, O_Y)
    # ic_ = 0.1*Noise2D().initial(num, O_X, O_Y, scaling = 1) # one cycle
    # ic = 0.1*(ic_-ic_[:,0,None,0,None]) + ic(grid_X, grid_Y)

    Soln_reno = SPDE2D(BC = 'P', IC = ic, mu = mu, sigma = sigma).Renormalization(0.1*W, O_T, O_X, O_Y) # generate through explicit scheme without renormalization
    Soln_expl = SPDE2D(BC = 'P', IC = ic, mu = mu, sigma = sigma).Parabolic(0.1*W, O_T, O_X, O_Y) # generate through explicit scheme with renormalization 

    return O_X, O_Y, O_T, W, eps, Soln_reno, Soln_expl

@hydra.main(version_base=None, config_path="../configs/", config_name="phi42")
def main(cfg: DictConfig):              
    np.random.seed(cfg.seed)
    O_X, O_Y, O_T, W, eps, soln_reno, soln_expl = solver(**cfg.sim)
    os.makedirs(cfg.save_dir, exist_ok=True)
    scipy.io.savemat(cfg.save_dir + 'Phi42+_reno_xi_u0_eps_{}_1200.mat'.format(cfg.sim.eps), mdict={'X':O_X, 'Y':O_Y, 'T':O_T, 'W': W, 'eps':eps, 'sol': soln_reno})
    scipy.io.savemat(cfg.save_dir + 'Phi42+_expl_xi_u0_eps_{}_1200.mat'.format(cfg.sim.eps), mdict={'X':O_X, 'Y':O_Y, 'T':O_T, 'W': W, 'eps':eps, 'sol': soln_expl})

if __name__ == "__main__":
    main()


