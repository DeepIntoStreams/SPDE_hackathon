import os
import hydra
from omegaconf import DictConfig
import scipy.io
import numpy as np
from data_gen.src.Noise import Noise, partition
from data_gen.src.general_solver import smooth_corr, general_1d_solver

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def simulator(a, b, Nx, s, t, Nt, num):
    dx, dt = (b-a)/Nx, (t-s)/Nt  # space-time increments
    O_X, O_T = partition(a,b,dx), partition(s,t,dt) # space grid O_X and time grid O_T

    u0 = np.array([[x * (1 - x) for x in np.linspace(a, b, Nx + 1)[:-1]] for _ in range(num)])  # initial condition

    # stochastic forcing
    r = 4  # Creates r/2 spatially smooth noise
    corr = lambda x, j, a: smooth_corr(x, j, a, r + 1.001)
    W_smooth = Noise().WN_space_time_many(s, t, dt * 0.1, a, b, dx, num, correlation=corr)
    W_smooth = W_smooth[:, ::10, :]

    L_kdv = [0, 0, 1e-3, -0.1, 0]
    mu_kdv = lambda x: 0
    sigma_kdv = lambda x: 1

    KdV, _, _ = general_1d_solver(L_kdv, u0, W_smooth[:, :, :-1], mu=mu_kdv, sigma=sigma_kdv, Burgers=-6)

    W = W_smooth.transpose(0,2,1)
    soln = KdV.transpose(0,2,1)
    return O_X, O_T, W, soln

@hydra.main(version_base=None, config_path="../configs/", config_name="KdV")
def main(cfg: DictConfig):
    np.random.seed(cfg.seed)
    O_X, O_T, W, soln = simulator(**cfg.sim)
    os.makedirs(cfg.save_dir, exist_ok=True)
    scipy.io.savemat(cfg.save_dir + 'kdv_xi_{}.mat'.format(cfg.sim.num), mdict={'X':O_X, 'T':O_T, 'W': W, 'sol': soln})


if __name__ == "__main__":
    main()