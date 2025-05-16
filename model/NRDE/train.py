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
import signatory
import wandb
import os
import os.path as osp
import sys
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(osp.join(current_directory, "..",".."))
from model.NRDE.NRDE import *
from model.utilities import *
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_training(data_path, ntrain, ntest, batch_size, epochs, learning_rate, weight_decay,
                 scheduler_step, scheduler_gamma, print_every,
                 dim_x, T, sub_t,
                 depth, window_length, normalizer, interpolation, solver,
                 hidden_channels, save_path):

    data = scipy.io.loadmat(data_path)

    O_X, O_T, W, Sol = data['X'], data['T'], data['W'], data['sol']
    xi = torch.from_numpy(W.astype(np.float32))
    data = torch.from_numpy(Sol.astype(np.float32))

    train_load, test_load, I, noise_size, norm = dataloader_nrde_1d(u=data, xi=xi, ntrain=ntrain,
                                                                    ntest=ntest, T=T, sub_t=sub_t,
                                                                    batch_size=batch_size, dim_x=dim_x,
                                                                    depth=depth, window_length=window_length,
                                                                    normalizer=normalizer,
                                                                    interpolation=interpolation)

    model = NeuralRDE(control_channels=noise_size, input_channels=dim_x,
                      hidden_channels=hidden_channels, output_channels=dim_x, interval=I,
                      interpolation=interpolation, solver=solver).cuda()
    print('The model has {} parameters'. format(count_params(model)))

    loss = LpLoss(size_average=False)
    model, losses_train, losses_test = train_nrde_1d(model, train_load, test_load, norm,
                                                    device, loss, batch_size=batch_size, epochs=epochs,
                                                    learning_rate=learning_rate, scheduler_step=scheduler_step,
                                                    scheduler_gamma=scheduler_gamma, print_every=print_every)

    torch.save(model.state_dict(), save_path)

def hyperparameter_tuning(data_path, ntrain, nval, ntest, batch_size, epochs, learning_rate,
                 plateau_patience, plateau_terminate, print_every,
                 dim_x, T, sub_t,
                 depth, window_length, normalizer, interpolation,
                 hidden_channels, solver, log_file, checkpoint_file, final_checkpoint_file):

    data = scipy.io.loadmat(data_path)

    O_X, O_T, W, Sol = data['X'], data['T'], data['W'], data['sol']
    xi = torch.from_numpy(W.astype(np.float32))
    data = torch.from_numpy(Sol.astype(np.float32))

    _, test_dl, I, noise_dim, norm = dataloader_nrde_1d(u=data, xi=xi, ntrain=ntrain + nval,
                                                        ntest=ntest, T=T, sub_t=sub_t, normalizer=normalizer,
                                                        depth=depth, window_length=window_length,
                                                        batch_size=batch_size, dim_x=dim_x, interpolation=interpolation)

    train_dl, val_dl, I, noise_dim, norm = dataloader_nrde_1d(u=data[:ntrain + nval], xi=xi[:ntrain + nval],
                                                              depth=depth, window_length=window_length,
                                                              ntrain=ntrain, ntest=nval,
                                                              T=T, sub_t=sub_t, normalizer=normalizer,
                                                              batch_size=batch_size, dim_x=dim_x, interpolation=interpolation)

    hyperparameter_search_nrde(train_dl, val_dl, test_dl, noise_dim, I, dim_x, norm,
                               hidden_channels, solver, epochs, print_every, learning_rate, plateau_patience,
                               plateau_terminate, log_file + '.csv', checkpoint_file, final_checkpoint_file)

@hydra.main(version_base=None, config_path="../config/", config_name="config_nrde_GL_u0_xi")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    run_training(**cfg.args)
    # hyperparameter_tuning(**cfg.tuning)
    print('Done.')


if __name__ == '__main__':
    main()