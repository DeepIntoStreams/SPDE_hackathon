import torch
import torch.optim as optim
import scipy.io
import wandb
import random
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from timeit import default_timer
import shutil
import os
import os.path as osp
import sys

current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(osp.join(current_directory, "..", ".."))
from FNO2D import *
from model.utilities import *
import warnings

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_xi(cfg):
    # Initialize a WandB experiment
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        config={**cfg.fixed.data, **cfg.fixed.model, **cfg.fixed.save}
    )
    run.name = f"{cfg.wandb.name}"
    cp_save_dir = f'{wandb.config.base_dir}/sweep_{run.sweep_id}/'
    os.makedirs(cp_save_dir, exist_ok=True)
    run.config.update({'save_dir': cp_save_dir}, allow_val_change=True)
    cp = f"{wandb.config.checkpoint_file}_run_{wandb.run.id}.pth"
    run.config.update({'checkpoint_file': cp_save_dir + cp}, allow_val_change=True)
    print("Current WandB Config:", dict(wandb.config))

    # Set random seed
    seed = wandb.config.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load data
    data = scipy.io.loadmat(wandb.config.data_path)
    if wandb.config.data_name == 'phi42':
        W, Sol = data['W'], data['sol']
        W = np.transpose(W, (0, 2, 3, 1))
        Sol = np.transpose(Sol, (0, 2, 3, 1))
    elif wandb.config.data_name == 'NS':
        W, Sol = data['forcing'], data['sol']
    print('data shape:')
    print(W.shape)
    print(Sol.shape)
    indices = np.random.permutation(Sol.shape[0])
    print('indices:', indices[:10])
    Sol = Sol[indices]
    W = W[indices]

    xi = torch.from_numpy(W.astype(np.float32))
    data = torch.from_numpy(Sol.astype(np.float32))
    ntrain = wandb.config.ntrain
    nval = wandb.config.nval
    ntest = wandb.config.ntest

    print('1 begin dl split')

    if cfg.wandb.name == 'xi':
        _, test_loader = dataloader_fno_2d_xi(u=data, xi=xi,
                                              ntrain=ntrain + nval,
                                              ntest=ntest,
                                              T=wandb.config.T,
                                              sub_t=wandb.config.sub_t,
                                              sub_x=wandb.config.sub_x,
                                              batch_size=wandb.config.batch_size)
        train_loader, val_loader = dataloader_fno_2d_xi(u=data[:ntrain + nval], xi=xi[:ntrain + nval],
                                                        ntrain=ntrain,
                                                        ntest=nval,
                                                        T=wandb.config.T,
                                                        sub_t=wandb.config.sub_t,
                                                        sub_x=wandb.config.sub_x,
                                                        batch_size=wandb.config.batch_size)
    elif cfg.wandb.name == 'u0':
        _, test_loader = dataloader_fno_2d_u0(u=data,
                                              ntrain=ntrain + nval,
                                              ntest=ntest,
                                              T=wandb.config.T,
                                              sub_t=wandb.config.sub_t,
                                              sub_x=wandb.config.sub_x,
                                              batch_size=wandb.config.batch_size)
        train_loader, val_loader = dataloader_fno_2d_u0(u=data[:ntrain + nval],
                                                        ntrain=ntrain,
                                                        ntest=nval,
                                                        T=wandb.config.T,
                                                        sub_t=wandb.config.sub_t,
                                                        sub_x=wandb.config.sub_x,
                                                        batch_size=wandb.config.batch_size)
    else:
        print("Unexpected cfg.wandb.name. Exiting...")
        sys.exit()

    model = FNO_space2D_time(modes1=wandb.config.modes1,
                             modes2=wandb.config.modes2,
                             modes3=wandb.config.modes3,
                             width=wandb.config.width,
                             L=wandb.config.L,
                             T=wandb.config.T // wandb.config.sub_t).cuda()
    print('The model has {} parameters'.format(count_params(model)))

    loss = LpLoss(size_average=False)

    _, _, _ = train_fno_2d_sweep(model, train_loader, val_loader, device, loss,
                                 batch_size=wandb.config.batch_size,
                                 epochs=wandb.config.epochs,
                                 learning_rate=wandb.config.learning_rate,
                                 weight_decay=wandb.config.weight_decay,
                                 plateau_patience=wandb.config.plateau_patience,
                                 plateau_terminate=wandb.config.plateau_terminate,
                                 delta=wandb.config.delta,
                                 print_every=wandb.config.print_every,
                                 checkpoint_file=wandb.config.checkpoint_file,
                                 wb=run, test_or_val='Val', tmp_loader=test_loader)
    model.load_state_dict(torch.load(wandb.config.checkpoint_file))
    # plot_2d_xi(model, test_loader, device, T=wandb.config.T // wandb.config.sub_t, wb=run)
    run.summary['loss_train'] = eval_fno_2d(model, train_loader, loss, wandb.config.batch_size, device)
    run.summary['loss_val'] = eval_fno_2d(model, val_loader, loss, wandb.config.batch_size, device)
    run.summary['loss_test'] = eval_fno_2d(model, test_loader, loss, wandb.config.batch_size, device)

    run.finish()


@hydra.main(version_base=None, config_path="../config/", config_name="sweep_fno_phi42_xi")
def main(cfg: DictConfig):
    sweep_cfg = OmegaConf.to_container(cfg.sweep, resolve=True)
    sweep_id = wandb.sweep(sweep_cfg, project=cfg.wandb.project)
    wandb.agent(sweep_id, function=lambda: train_xi(cfg))
    save_model(cfg, sweep_id)


if __name__ == '__main__':
    main()
