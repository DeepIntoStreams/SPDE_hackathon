import torch
import torch.optim as optim
import scipy.io
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from timeit import default_timer
import os
import os.path as osp
import sys

current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(osp.join(current_directory, "..", ".."))
from model.DeepONet.deepOnet import *
from model.utilities import *
import warnings

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(cfg):
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        # name=cfg.wandb.name,
        config={**cfg.fixed.data,
                **cfg.fixed.train,
                **cfg.fixed.save}
    )
    run.name = f"{cfg.wandb.name}-{wandb.config.width}-{wandb.config.branch_depth}-{wandb.config.trunk_depth}"

    cp_save_dir = f'{wandb.config.base_dir}/sweep-{run.sweep_id}/'
    os.makedirs(cp_save_dir, exist_ok=True)
    run.config.update({'save_dir': cp_save_dir}, allow_val_change=True)
    cp = f"{wandb.config.checkpoint_file}_run-{wandb.run.id}.pth"
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

    data = scipy.io.loadmat(wandb.config.data_path)

    W, Sol = data['W'], data['sol']
    print('W shape:', W.shape)
    print('Sol shape:', Sol.shape)
    xi = torch.from_numpy(W.astype(np.float32))
    data = torch.from_numpy(Sol.astype(np.float32))

    ntrain = wandb.config.ntrain
    nval = wandb.config.nval
    ntest = wandb.config.ntest

    if cfg.task == 'xi':
        _, test_dl, norm, grid = dataloader_deeponet_1d_xi(u=data, xi=xi,
                                                           ntrain=ntrain + nval,
                                                           ntest=ntest,
                                                           T=wandb.config.T,
                                                           sub_t=wandb.config.sub_t,
                                                           batch_size=wandb.config.batch_size,
                                                           dim_x=wandb.config.dim_x,
                                                           normalizer=wandb.config.normalizer,)

        train_dl, val_dl, norm, grid = dataloader_deeponet_1d_xi(u=data[:ntrain + nval],
                                                                 xi=xi[:ntrain + nval],
                                                                 ntrain=ntrain, ntest=nval,
                                                                 T=wandb.config.T,
                                                                 sub_t=wandb.config.sub_t,
                                                                 batch_size=wandb.config.batch_size,
                                                                 dim_x=wandb.config.dim_x,
                                                                 normalizer=wandb.config.normalizer,)

        S = wandb.config.dim_x * (wandb.config.T - 1)
    elif cfg.task == 'u0':
        print('There is no known forcing!')
        _, test_dl, norm, grid = dataloader_deeponet_1d_u0(u=data,
                                                           ntrain=ntrain + nval,
                                                           ntest=ntest,
                                                           T=wandb.config.T,
                                                           sub_t=wandb.config.sub_t,
                                                           batch_size=wandb.config.batch_size,
                                                           dim_x=wandb.config.dim_x,
                                                           normalizer=wandb.config.normalizer,)

        train_dl, val_dl, norm, grid = dataloader_deeponet_1d_u0(u=data[:ntrain + nval],
                                                                 ntrain=ntrain, ntest=nval,
                                                                 T=wandb.config.T,
                                                                 sub_t=wandb.config.sub_t,
                                                                 batch_size=wandb.config.batch_size,
                                                                 dim_x=wandb.config.dim_x,
                                                                 normalizer=wandb.config.normalizer,)
        S = wandb.config.dim_x

    branch = [S] + wandb.config.branch_depth * [wandb.config.width]
    trunk = [grid.shape[-1]] + wandb.config.trunk_depth * [wandb.config.width]
    model = DeepONetCP(branch_layer=branch,
                       trunk_layer=trunk).to(device)

    print('The model has {} parameters'.format(count_params(model)))

    loss = LpLoss(size_average=False)

    _, _, _ = train_deepOnet_1d_wb(model, train_dl, val_dl,
                                   grid, norm, device, loss,
                                   batch_size=wandb.config.batch_size,
                                   epochs=wandb.config.epochs,
                                   learning_rate=wandb.config.learning_rate,
                                   plateau_patience=wandb.config.plateau_patience,
                                   factor=wandb.config.factor,
                                   plateau_terminate=wandb.config.plateau_terminate,
                                   delta=wandb.config.delta,
                                   print_every=wandb.config.print_every,
                                   checkpoint_file=wandb.config.checkpoint_file,
                                   wb=run, test_or_val='Val')

    model.load_state_dict(torch.load(wandb.config.checkpoint_file))
    run.summary['loss_train'] = eval_deeponet(model, train_dl, loss, wandb.config.batch_size, device, grid, norm)
    run.summary['loss_val'] = eval_deeponet(model, val_dl, loss, wandb.config.batch_size, device, grid, norm)
    run.summary['loss_test'] = eval_deeponet(model, test_dl, loss, wandb.config.batch_size, device, grid, norm)

    run.finish()


@hydra.main(version_base=None, config_path="../config/", config_name="deeponet")
def main(cfg: DictConfig):
    sweep_cfg = OmegaConf.to_container(cfg.sweep, resolve=True)
    sweep_id = wandb.sweep(sweep_cfg, project=cfg.wandb.project)
    wandb.agent(sweep_id, function=lambda: train(cfg))
    save_model(cfg, sweep_id)


if __name__ == '__main__':
    main()
