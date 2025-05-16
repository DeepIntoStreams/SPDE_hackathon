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
import os
import os.path as osp
import sys
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(osp.join(current_directory, "..",".."))
from model.NSPDE.neural_spde import *
from model.NSPDE.utilities import *
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_training(data_path, ntrain, ntest, batch_size, sub_x, T, sub_t,
                 hidden_channels, n_iter, modes1, modes2, solver,
                 epochs, learning_rate, scheduler_step, scheduler_gamma,
                 print_every, save_path):

    # Load data.
    data = scipy.io.loadmat(data_path)
    W, Sol = data['forcing'], data['sol']
    xi = torch.from_numpy(W.astype(np.float32))
    data = torch.from_numpy(Sol.astype(np.float32))

    train_loader, test_loader = dataloader_nspde_2d(u=data, xi=xi, ntrain=ntrain,
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
    # Load data.
    data = scipy.io.loadmat(data_path)
    W, Sol = data['forcing'], data['sol']
    xi = torch.from_numpy(W.astype(np.float32))
    data = torch.from_numpy(Sol.astype(np.float32))

    _, test_dl = dataloader_nspde_2d(u=data, xi=xi, ntrain=ntrain+nval,
                                         ntest=ntest, T=T, sub_t=sub_t, sub_x=sub_x,
                                         batch_size=batch_size)

    train_dl, val_dl = dataloader_nspde_2d(u=data[:ntrain + nval], xi=xi[:ntrain + nval],
                                                   ntrain=ntrain, ntest=nval, T=T, sub_t=sub_t, sub_x=sub_x,
                                                   batch_size=batch_size)

    hyperparameter_search_nspde_2d(train_dl, val_dl, test_dl, solver=solver,
                                   d_h=hidden_channels, iter=n_iter, modes1=modes1, modes2=modes2,
                                   epochs=epochs, print_every=print_every, lr=learning_rate,
                                   plateau_patience=plateau_patience, plateau_terminate=plateau_terminate,
                                   log_file=log_file + '.csv', checkpoint_file=checkpoint_file,
                                   final_checkpoint_file=final_checkpoint_file)


@hydra.main(version_base=None, config_path="../config/", config_name="config_nspde_NS_xi.yaml")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    # run_training(**cfg.args)
    hyperparameter_tuning(**cfg.tuning)
    print('Done.')


if __name__ == '__main__':
    main()