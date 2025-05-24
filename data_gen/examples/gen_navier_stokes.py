# Adapted from https://github.com/crispitagorico/torchspde
# Modified for current implementation by the authors of SPDEBench


import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import math
import scipy.io
import random
import os
import os.path as osp
import sys
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(osp.join(current_directory, "..", ".."))
from data_gen.src.generator_sns import navier_stokes_2d
from data_gen.src.random_forcing import GaussianRF
from timeit import default_timer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def simulator(cfg):

    # Set up 2d GRF with covariance parameters
    GRF = GaussianRF(2, cfg.s, alpha=cfg.alpha, tau=cfg.tau, device=device)

    # Forcing function: 0.1*(sin(2pi(x+y)) + cos(2pi(x+y)))
    t = torch.linspace(0, 1, cfg.s + 1, device=device)
    t = t[0:-1]
    X, Y = torch.meshgrid(t, t)
    f = 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))

    # Stochastic forcing function: sigma*dW/dt
    stochastic_forcing = {'alpha': cfg.alpha_Q, 'kappa': cfg.kappa, 'sigma': cfg.sigma, 'truncation': cfg.truncation}

    # Number of snapshots from solution
    T = cfg.T
    record_steps = int(T / (cfg.delta_t))

    # Prepare save_dir
    ic_type = 'xi' if cfg.fix_u0 else 'u0_xi'
    folder = f'{cfg.save_dir}NS_{ic_type}_trc{cfg.truncation}/'
    os.makedirs(folder, exist_ok=True)

    # Solve equations in batches (order of magnitude speed-up)
    c = 0
    t0 = default_timer()
    bsize = cfg.bsize
    sub_x = cfg.sub_x
    sub_t = cfg.sub_t
    # Sample random fields
    if cfg.fix_u0 == True:
        w0 = GRF.sample(1).repeat(bsize,1,1)
    else:
        w_star = GRF.sample(1).repeat(bsize, 1, 1)

    for j in range(cfg.N // bsize):
        if cfg.fix_u0 == False:
            w0 = w_star + GRF.sample(bsize)

        sol, sol_t, forcing = navier_stokes_2d([1, 1], w0, f, cfg.nu, T, cfg.delta_t, record_steps, stochastic_forcing)

        # add time 0
        time = torch.zeros(record_steps + 1)
        time[1:] = sol_t.cpu()
        sol = torch.cat([w0[..., None], sol], dim=-1)
        forcing = torch.cat([torch.zeros_like(w0)[..., None], forcing], dim=-1)  # [bsize, x, y, T/delta_t + 1]

        c += bsize
        t1 = default_timer()
        print(j, c, t1 - t0)

        # sum forcing in each time period
        forcing = forcing[..., :sub_t * ((record_steps + 1) // sub_t)]
        forcing = forcing.view(*forcing.shape[:-1], (record_steps + 1) // sub_t, sub_t)  # [..., T_steps//sub_t, sub_t]
        summed_forcing = forcing.sum(dim=-1)  # [bsize, x, y, T_steps//sub_t]

        scipy.io.savemat(folder + f'NS_small_{j}.mat',
                         mdict={'t': time.numpy(),
                                'sol': sol[:, ::sub_x, ::sub_x, ::sub_t].cpu().numpy(),  # [bsize, x, y, T_steps//sub_t + 1]
                                'forcing': summed_forcing[:, ::sub_x, ::sub_x, :].cpu().numpy(),  # [bsize, x, y, T_steps//sub_t]
                                'param': stochastic_forcing})

    return folder


def merge(N, bsize, save_dir, output_name = "merged_ns.mat"):

    n_files = N // bsize  # number of files

    sol_list = []
    forcing_list = []
    t = None
    param = None

    # Extract data from original files
    for j in range(n_files):
        data = scipy.io.loadmat(f"{save_dir}NS_small_{j}.mat")
        if j == 0:
            t = data['t']
            param = data['param']
        sol_list.append(data['sol'])
        forcing_list.append(data['forcing'])

    merged_sol = np.concatenate(sol_list, axis=0)
    merged_forcing = np.concatenate(forcing_list, axis=0)

    merged_data = {
        't': t,
        'sol': merged_sol,
        'forcing': merged_forcing,
        'param': param
    }

    output_path = save_dir + output_name
    scipy.io.savemat(output_path, merged_data, do_compression=True)
    print(f"Successfully merged! Saved to:" + output_path)

    print("sol shape:", merged_sol.shape)  # (n_files*batch_size, dim_x, dim_y, dim_t)
    print("forcing shape:", merged_forcing.shape)
    print("t", t)
    print("param:", param)

@hydra.main(version_base=None, config_path="../configs/", config_name="NS")
def main(cfg: DictConfig):
    seed = cfg.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    folder = simulator(cfg)
    if cfg.merge_data:
        merge(cfg.N, cfg.bsize, folder, output_name = f"merged_ns_{cfg.N}.mat")


if __name__ == "__main__":
    main()