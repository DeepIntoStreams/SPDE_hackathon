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
from FNO1D import *
from model.utilities import *
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_training_xi(data_path, ntrain, ntest, batch_size, epochs, learning_rate,
                 scheduler_step, scheduler_gamma, print_every,
                 dim_x, T, sub_t, dataset,
                 modes1, modes2, width, L,
                 save_path):

    data = scipy.io.loadmat(data_path)

    O_X, O_T, W, Sol = data['X'], data['T'], data['W'], data['sol']
    xi = torch.from_numpy(W.astype(np.float32))
    data = torch.from_numpy(Sol.astype(np.float32))

    train_loader, test_loader = dataloader_fno_1d_xi(data, xi, ntrain,
                                                     ntest, T, sub_t, batch_size,
                                                     dim_x, dataset)

    model = FNO_space1D_time(modes1, modes2, width, T=T, L=L).cuda()

    print('The model has {} parameters'. format(count_params(model)))

    loss = LpLoss(size_average=False)

    model, losses_train, losses_test = train_fno_1d(model, train_loader, test_loader,
                                                    device, loss, batch_size=batch_size, epochs=epochs,
                                                    learning_rate=learning_rate, scheduler_step=scheduler_step,
                                                    scheduler_gamma=scheduler_gamma, print_every=print_every)

    torch.save(model.state_dict(), save_path)

def run_training_u0(data_path, ntrain, ntest, batch_size, epochs, learning_rate,
                 scheduler_step, scheduler_gamma, print_every,
                 dim_x, T, sub_t, dataset,
                 modes1, modes2, width, L,
                 save_path):

    data = scipy.io.loadmat(data_path)

    O_X, O_T, W, Sol = data['X'], data['T'], data['W'], data['sol']
    data = torch.from_numpy(Sol.astype(np.float32))

    train_loader, test_loader = dataloader_fno_1d_u0(data, ntrain,
                                                     ntest, T, sub_t, batch_size,
                                                     dim_x, dataset)

    model = FNO_space1D_time(modes1, modes2, width, L, T).cuda()

    print('The model has {} parameters'. format(count_params(model)))

    loss = LpLoss(size_average=False)

    model, losses_train, losses_test = train_fno_1d(model, train_loader, test_loader,
                                                    device, loss, batch_size=batch_size, epochs=epochs,
                                                    learning_rate=learning_rate, scheduler_step=scheduler_step,
                                                    scheduler_gamma=scheduler_gamma, print_every=print_every)

    torch.save(model.state_dict(), save_path)

def hyperparameter_tuning(data_path, ntrain, nval, ntest, batch_size, epochs, learning_rate,
                 plateau_patience, plateau_terminate, print_every,
                 dim_x, T, sub_t,
                 d_h, modes1, modes2, iter,
                 log_file, checkpoint_file, final_checkpoint_file):

    data = scipy.io.loadmat(data_path)

    O_X, O_T, W, Sol = data['X'], data['T'], data['W'], data['sol']
    xi = torch.from_numpy(W.astype(np.float32))
    data = torch.from_numpy(Sol.astype(np.float32))

    _, test_dl = dataloader_fno_1d_xi(u=data, xi=xi, ntrain=ntrain + nval,
                                      ntest=ntest, T=T, sub_t=sub_t,
                                      batch_size=batch_size, dim_x=dim_x)

    train_dl, val_dl = dataloader_fno_1d_xi(u=data[:ntrain + nval], xi=xi[:ntrain + nval],
                                            ntrain=ntrain, ntest=nval, T=T, sub_t=sub_t,
                                            batch_size=batch_size, dim_x=dim_x)

    hyperparameter_search_fno1d(train_dl, val_dl, test_dl, T=T,
                                d_h=d_h, iter=iter, modes1=modes1, modes2=modes2,
                                lr=learning_rate, epochs=epochs, print_every=print_every, plateau_patience=plateau_patience,
                                plateau_terminate=plateau_terminate, log_file=log_file + '.csv',
                                checkpoint_file=checkpoint_file,
                                final_checkpoint_file=final_checkpoint_file)


@hydra.main(version_base=None, config_path="../config/", config_name="config_fno_GL_xi")
def main(cfg: DictConfig):
    print('Running FNO wave xi->u  ...')
    # run_training_xi(**cfg.args)
    # run_training_u0(**cfg.args)
    hyperparameter_tuning(**cfg.tuning)
    print('Done.')


if __name__ == '__main__':
    main()
