import hydra
from omegaconf import DictConfig
import torch
import numpy as np
from numpy import matlib
import math
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import scipy.io
import h5py
import pickle
import matplotlib.animation
import os
import random
plt.rcParams["animation.html"] = "jshtml"
from data_gen.src.generator_sns import navier_stokes_2d
from data_gen.src.random_forcing import GaussianRF
from timeit import default_timer
from data_gen.src.random_forcing import GaussianRF, get_twod_bj, get_twod_dW
seed=42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def simulator_xi(save_dir, seed, nu, alpha, tau, stochastic_forcing, s, T, delta_t, N, bsize):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')

    os.makedirs(save_dir, exist_ok=True)

    # Set up 2d GRF with covariance parameters
    GRF = GaussianRF(2, s, alpha=alpha, tau=tau, device=device)

    # Forcing function: 0.1*(sin(2pi(x+y)) + cos(2pi(x+y)))
    t = torch.linspace(0, 1, s + 1, device=device)
    t = t[0:-1]
    X, Y = torch.meshgrid(t, t)
    f = 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))

    # Number of snapshots from solution
    record_steps = int(T / (delta_t))

    # Solve equations in batches (order of magnitude speed-up)
    c = 0
    t0 = default_timer()
    # Sample random fields
    # w0 = GRF.sample(1).repeat(bsize,1,1)
    for j in range(N // bsize):
        # initial condition
        w0 = GRF.sample(bsize)

        # compute the numerical solution
        sol, sol_t, forcing = navier_stokes_2d([1, 1], w0, f, nu, T, delta_t, record_steps, stochastic_forcing)
        print("sol.shape = ", sol.shape)
        print("forcing.shape = ", forcing.shape)

        # add time 0
        time = torch.zeros(record_steps + 1)
        time[1:] = sol_t.cpu()
        sol = torch.cat([w0[..., None], sol], dim=-1)
        forcing = torch.cat([torch.zeros_like(w0)[..., None], forcing], dim=-1)

        c += bsize
        t1 = default_timer()
        print(j, c, t1 - t0)

        scipy.io.savemat(save_dir + 'today_my_restru_u0xi_{}.mat'.format(j),
                         mdict={'t': time.numpy(), 'sol': sol.cpu().numpy(), 'forcing': forcing.cpu().numpy(),
                                'param': stochastic_forcing})
        # scipy.io.savemat(save_dir+'testAAA_ns_xi_small_{}.mat'.format(j),
        #                  mdict={'t': time[::10].numpy(), 'sol': sol[:,::1,::1,::10].cpu().numpy(),
        #                         'forcing': forcing[:,::1,::1,::10].cpu().numpy(), 'param':stochastic_forcing})

def simulator_u0_xi(save_dir, seed, nu, alpha, tau, stochastic_forcing, s, T, delta_t, N, bsize):
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(save_dir, exist_ok=True)

    # Set up 2d GRF with covariance parameters
    GRF = GaussianRF(2, s, alpha=alpha, tau=tau, device=device)

    # Forcing function: 0.1*(sin(2pi(x+y)) + cos(2pi(x+y)))
    t = torch.linspace(0, 1, s + 1, device=device)
    t = t[0:-1]
    X, Y = torch.meshgrid(t, t)
    f = 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))

    # Number of snapshots from solution
    record_steps = int(T / (delta_t))

    # Solve equations in batches (order of magnitude speed-up)
    c = 0
    t0 = default_timer()
    # Sample random fields
    # w0 = GRF.sample(1).repeat(bsize,1,1)
    for j in range(N // bsize):
        w0 = GRF.sample(bsize)

        sol, sol_t, forcing = navier_stokes_2d([1, 1], w0, f, nu, T, delta_t, record_steps, stochastic_forcing)

        # add time 0
        time = torch.zeros(record_steps + 1)
        time[1:] = sol_t.cpu()
        sol = torch.cat([w0[..., None], sol], dim=-1)
        forcing = torch.cat([torch.zeros_like(w0)[..., None], forcing], dim=-1)

        c += bsize
        t1 = default_timer()
        print(j, c, t1 - t0)

        scipy.io.savemat(save_dir + '{}.mat'.format(j),
                         mdict={'t': time.numpy(), 'sol': sol.cpu().numpy(), 'forcing': forcing.cpu().numpy(),
                                'param': stochastic_forcing})
        scipy.io.savemat(save_dir+'small_{}.mat'.format(j), mdict={'t': time[::4].numpy(), 'sol': sol[:,::4,::4,::4].cpu().numpy(), 'forcing': forcing[:,::4,::4,::4].cpu().numpy(), 'param':stochastic_forcing})


@hydra.main(version_base=None, config_path="../configs/", config_name="navier_stokes_xi")
def main(cfg: DictConfig):
    simulator_xi(**cfg)
    print('Done.')


if __name__ == "__main__":
    main()