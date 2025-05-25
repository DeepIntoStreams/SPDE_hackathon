import torch
import torch.optim as optim
import scipy.io
import random
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from timeit import default_timer
import os
import os.path as osp
import sys
from model.NSPDE.neural_aeps_spde import *
from model.NSPDE.utilities_aeps import *
from model.NSPDE.Noise import Noise2D
from model.NSPDE.SPDEs2D import SPDE2D
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_training(data_path, ntrain, ntest, batch_size, sub_x, T, sub_t,
                 hidden_channels, n_iter, modes1, modes2, solver,
                 epochs, learning_rate, scheduler_step, scheduler_gamma,
                 print_every, save_path):
    
    # Load data
    data = scipy.io.loadmat(data_path)
    O_X, O_Y, O_T, W, Sol, eps =data['X'].squeeze(), data['Y'].squeeze(), data['T'].squeeze(), data['W'], data['sol'], data['eps'].squeeze()
    xi = torch.from_numpy(W.astype(np.float32))
    data = torch.from_numpy(Sol.astype(np.float32))
    
    X_eps = SPDE2D().FT_solver(W, O_T, O_X, O_Y)
    a_eps = SPDE2D().MC(X_eps)
    a_eps = torch.from_numpy(a_eps).unsqueeze(0).repeat(1200, 1, 1, 1)

    train_loader, test_loader = dataloader_nspde_2d(u=data, xi=xi, a_eps=a_eps, ntrain=ntrain,
                                                    ntest=ntest, T=T, sub_t=sub_t, sub_x=sub_x,
                                                    batch_size=batch_size)

    # Define the model.
    model = NeuralSPDE(dim=2, in_channels=1, noise_channels=1, hidden_channels=hidden_channels,
                       n_iter=n_iter, modes1=modes1, modes2=modes2, solver=solver).cuda()

    print('The model has {} parameters'.format(count_params(model)))

    # Train the model.
    loss = LpLoss(size_average=False)
    model, losses_train, losses_test = train_nspde(model, train_loader, test_loader, device, loss,
                                                   batch_size=batch_size, epochs=epochs,
                                                   learning_rate=learning_rate, scheduler_step=scheduler_step,
                                                   scheduler_gamma=scheduler_gamma, print_every=print_every)

    torch.save(model.state_dict(), save_path)

    plt.plot(np.arange(1, len(losses_train) * 5, 5), losses_train, label='train')
    plt.plot(np.arange(1, len(losses_test) * 5, 5), losses_test, label='test')
    plt.xlabel('Epoch')
    plt.ylabel('Relative L2 loss')
    plt.legend()
    plt.show()


def hyperparameter_tuning(data_path, ntrain, nval, ntest, batch_size, epochs, learning_rate,
                 plateau_patience, plateau_terminate, print_every,
                 sub_x, T, sub_t,
                 hidden_channels, n_iter, modes1, modes2, solver,
                 log_file, checkpoint_file, final_checkpoint_file):
    # Load data
    data = scipy.io.loadmat(data_path)
    O_X, O_Y, O_T, W, Sol, eps =data['X'].squeeze(), data['Y'].squeeze(), data['T'].squeeze(), data['W'], data['sol'], data['eps'].squeeze()
    xi = torch.from_numpy(W.astype(np.float32))
    data = torch.from_numpy(Sol.astype(np.float32))
    
    X_eps = SPDE2D().FT_solver(W, O_T, O_X, O_Y)
    a_eps = SPDE2D().MC(X_eps)
    a_eps = torch.from_numpy(a_eps).unsqueeze(0).repeat(1200, 1, 1, 1)

    _, test_dl = dataloader_nspde_2d(u=data, xi=xi, a_eps=a_eps, ntrain=ntrain+nval,
                                         ntest=ntest, T=T, sub_t=sub_t, sub_x=sub_x,
                                         batch_size=batch_size)

    train_dl, val_dl = dataloader_nspde_2d(u=data[:ntrain + nval], xi=xi[:ntrain + nval], a_eps=a_eps[:ntrain + nval],
                                                   ntrain=ntrain, ntest=nval, T=T, sub_t=sub_t, sub_x=sub_x,
                                                   batch_size=batch_size)

    hyperparameter_search_nspde_2d(train_dl, val_dl, test_dl, solver=solver,
                                   d_h=hidden_channels, iter=n_iter, modes1=modes1, modes2=modes2,
                                   epochs=epochs, print_every=print_every, lr=learning_rate,
                                   plateau_patience=plateau_patience, plateau_terminate=plateau_terminate,
                                   log_file=log_file + '.csv', checkpoint_file=checkpoint_file,
                                   final_checkpoint_file=final_checkpoint_file)


@hydra.main(version_base=None, config_path="../config/", config_name="nspde_phi42.yaml")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # Set random seed
    seed = cfg.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # run_training(**cfg.args)
    hyperparameter_tuning(**cfg.tuning)


if __name__ == '__main__':
    main()