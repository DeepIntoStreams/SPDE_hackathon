import torch
import torch.optim as optim
import scipy.io
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from timeit import default_timer
import random
import os
import os.path as osp
import sys

current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(osp.join(current_directory, "..", ".."))
from model.NCDEFNO.NCDEFNO_1D import *
from model.utilities import *
import warnings

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(cfg):
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        config={**cfg.fixed.data,
                **cfg.fixed.model,
                **cfg.fixed.train,
                **cfg.fixed.save}
    )

    os.makedirs(wandb.config.base_dir, exist_ok=True)
    cp = f"{wandb.config.checkpoint_file}_run-{wandb.run.id}.pth"
    run.config.update({'checkpoint_file': f'{wandb.config.base_dir}/{cp}'}, allow_val_change=True)

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

    _, test_dl = dataloader_ncdeinf_1d(u=data, xi=xi,
                                       ntrain=ntrain + nval,
                                       ntest=ntest,
                                       T=wandb.config.T,
                                       sub_t=wandb.config.sub_t,
                                       batch_size=wandb.config.batch_size,
                                       dim_x=wandb.config.dim_x,
                                       interpolation=wandb.config.interpolation)

    train_dl, val_dl = dataloader_ncdeinf_1d(u=data[:ntrain + nval],
                                             xi=xi[:ntrain + nval],
                                             ntrain=ntrain, ntest=nval,
                                             T=wandb.config.T,
                                             sub_t=wandb.config.sub_t,
                                             batch_size=wandb.config.batch_size,
                                             dim_x=wandb.config.dim_x,
                                             interpolation=wandb.config.interpolation)

    model = NeuralCDE(data_size=wandb.config.data_size,
                      noise_size=wandb.config.noise_size,
                      hidden_channels=wandb.config.hidden_channels,
                      output_channels=wandb.config.output_channels,
                      interpolation=wandb.config.interpolation,
                      solver=wandb.config.solver).cuda()

    print('The model has {} parameters'.format(count_params(model)))

    loss = LpLoss(size_average=False)

    _, _, _ = train_ncdeinf_1d_wb(model, train_dl, val_dl, device, loss,
                                  batch_size=wandb.config.batch_size,
                                  epochs=wandb.config.epochs,
                                  learning_rate=wandb.config.learning_rate,
                                  # plateau_patience=wandb.config.plateau_patience,
                                  factor=wandb.config.factor,
                                  plateau_terminate=wandb.config.plateau_terminate,
                                  delta=wandb.config.delta,
                                  print_every=wandb.config.print_every,
                                  checkpoint_file=wandb.config.checkpoint_file,
                                  wb=run, test_or_val='Val')

    model.load_state_dict(torch.load(wandb.config.checkpoint_file))
    run.summary['loss_train'] = eval_ncdeinf_1d(model, train_dl, loss, wandb.config.batch_size, device)
    run.summary['loss_val'] = eval_ncdeinf_1d(model, val_dl, loss, wandb.config.batch_size, device)
    run.summary['loss_test'] = eval_ncdeinf_1d(model, test_dl, loss, wandb.config.batch_size, device)

    run.finish()


@hydra.main(version_base=None, config_path="../config/", config_name="ncde-fno")
def main(cfg: DictConfig):
    train(cfg)


if __name__ == '__main__':
    main()
