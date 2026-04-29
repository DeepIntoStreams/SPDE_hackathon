# Portions of this code adapted from https://github.com/crispitagorico/torchspde
# Modified for current implementation by the authors of SPDEBench

import hydra
from omegaconf import DictConfig, OmegaConf
import scipy.io
import numpy as np
import os
import os.path as osp
import sys
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(osp.join(current_directory, "..",".."))
from data_gen.src.NoiseND import NoiseND
from data_gen.src.general_solver import general_1d_solver


# smooth Q noise as in Example 10.8 of `An Introduction to Computational Stochastic PDEs' by Lord, Powell & Shardlow
def smooth_corr( modes, lengths, r):
    j = modes[0]
    if j == 0:
        return 0.0
    return (j // 2 + 1) ** (-r)

def simulator(a, b, Nx, s, t, Nt, noise_type, sigma, truncation, fix_u0, num):
    noise = NoiseND()
    dx, dt = (b-a)/Nx, (t-s)/Nt  # space-time increments
    O_X = noise.partition_axis(a, b, dx)
    O_T = noise.partition_axis(s, t, dt)

    u0 = np.array([[np.sin(2*np.pi*x) for x in np.linspace(a, b, Nx + 1)[:-1]] for _ in range(num)])  # initial condition
    if not fix_u0:  # varying initial condition
        X_ = np.linspace(-0.5, 0.5, Nx + 1)
        ic_ = noise.initial(num, (X_,))[..., :-1]
        u0 = (ic_ - ic_[:, 0, None]) + u0
        print("u0 is varying!")
    else:
        print("u0 is fixed!")

    # stochastic forcing
    if noise_type == 'Q':
        r = 4  # Creates r/2 spatially smooth noise
        corr = lambda modes, lengths: smooth_corr(modes, lengths, r + 1.001)
        noise_q = NoiseND(covariance='q_wiener', q_spectrum=corr)
        W_smooth = noise_q.WN_space_time_KPZ(
            s, t, dt * 0.1,
            bounds=((a, b),),
            steps=(dx,),
            num=num,
            truncation=(truncation + 1,),
        )
        W_smooth = W_smooth[:, ::10, :]
    elif noise_type == 'cyl':
        W_smooth = noise.WN_space_time(
            s,
            t,
            dt,
            bounds=((a, b),),
            steps=(dx,),
            num=num,
            truncation=(truncation + 1,),
        )
    else:
        print('Invalid noise type!')
        exit(0)

    L_kdv = [0, 0, 0, -0.1, 0]
    mu_kdv = lambda x: 0
    sigma_kdv = lambda x: sigma

    KdV, _, _ = general_1d_solver(
        L_kdv,
        u0,
        W_smooth[:, :, :-1],
        mu=mu_kdv,
        sigma=sigma_kdv,
        T=t - s,
        X=b - a,
        Burgers=6,
    )

    W = W_smooth.transpose(0,2,1)
    soln = KdV.transpose(0,2,1)
    return O_X, O_T, W, soln

@hydra.main(version_base=None, config_path="../configs/", config_name="KdV")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))
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
