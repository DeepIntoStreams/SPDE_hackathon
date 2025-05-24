import hydra
from omegaconf import DictConfig
import scipy.io
import numpy as np
import os
import os.path as osp
import sys
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(osp.join(current_directory, "..", ".."))
from data_gen.src.Noise2D import Noise2D
from data_gen.src.SPDEs2D import SPDE2D

def solver(a, b, Nx, c, d, Ny, s, t, Nt, num, eps, sigma, fix_u0):

    dx, dy, dt = (b-a)/Nx, (d-c)/Ny, (t - s) / Nt  # space-time increments

    ic = lambda x, y: np.sin(2 * np.pi * (x + y)) + np.cos(2 * np.pi * (x + y)) # initial condition (fixed)

    mu = lambda x: 3*x-x**3 # drift
    # sigma_fun = lambda x: sigma # additive diffusive term

    O_X, O_Y = Noise2D().partition_2d(a,b,dx,c,d,dy) # space grid O_X, O_Y
    O_T = Noise2D().partition(s,t,dt) # time grid O_T
    W = Noise2D().WN_space_time_2d_many(s, t, dt, a, b, dx, c, d, dy, num, eps, eps) # create realizations of space-time white noise

    if not fix_u0: # varying initial condition
        grid_X, grid_Y = np.meshgrid(O_X, O_Y)
        ic_ = 0.1*Noise2D().initial(num, O_X, O_Y, scaling = 1) # one cycle
        ic = 0.1*(ic_-ic_[:,0,None,0,None]) + ic(grid_X, grid_Y)
        print("u0 is varying!")
    else:
        print("u0 is fixed!")

    Soln_reno = SPDE2D(BC = 'P', IC = ic, mu = mu, sigma = sigma).Renormalization(W, O_T, O_X, O_Y) # generate through explicit scheme without renormalization
    Soln_expl = SPDE2D(BC = 'P', IC = ic, mu = mu, sigma = sigma).Parabolic(W, O_T, O_X, O_Y) # generate through explicit scheme with renormalization

    return O_X, O_Y, O_T, W, eps, Soln_reno, Soln_expl

@hydra.main(version_base=None, config_path="../configs/", config_name="phi42")
def main(cfg: DictConfig):              
    np.random.seed(cfg.seed)
    O_X, O_Y, O_T, W, eps, soln_reno, soln_expl = solver(**cfg.sim)

    os.makedirs(cfg.save_dir, exist_ok=True)
    ic_type = 'xi' if cfg.sim.fix_u0 else 'xi_u0'

    reno_filename = f'{cfg.save_name}_reno_{ic_type}_eps{cfg.sim.eps}_{cfg.sim.num}.mat'
    scipy.io.savemat(cfg.save_dir + reno_filename, mdict={'X':O_X, 'Y':O_Y, 'T':O_T, 'W': W, 'eps':eps, 'sol': soln_reno})
    print("Saved to", cfg.save_dir + reno_filename)

    expl_filename = f'{cfg.save_name}_expl_{ic_type}_eps{cfg.sim.eps}_{cfg.sim.num}.mat'
    scipy.io.savemat(cfg.save_dir + expl_filename, mdict={'X':O_X, 'Y':O_Y, 'T':O_T, 'W': W, 'eps':eps, 'sol': soln_expl})
    print("Saved to", cfg.save_dir + expl_filename)

if __name__ == "__main__":
    main()


