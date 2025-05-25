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

def run_training(data_path_2, data_path_8, data_path_32, data_path_64, data_path_128, ntrain, ntest, batch_size, sub_x, T, sub_t,
                 hidden_channels, n_iter, modes1, modes2, solver,
                 epochs, learning_rate, scheduler_step, scheduler_gamma,
                 print_every, save_path):

    # Load data
    data_2 = scipy.io.loadmat(data_path_2)
    data_8 = scipy.io.loadmat(data_path_8)
    data_32 = scipy.io.loadmat(data_path_32)
    data_64 = scipy.io.loadmat(data_path_64)
    data_128 = scipy.io.loadmat(data_path_128)

    O_X, O_Y, O_T, W_2, Sol_2=data_2['X'].squeeze(), data_2['Y'].squeeze(), data_2['T'].squeeze(), data_2['W'][:240,:,:,:], data_2['sol'][:240,:,:,:]
    X_2 = SPDE2D().FT_solver(W_2, O_T, O_X, O_Y)
    a_2 = SPDE2D().MC(X_2)
    a_2 = torch.from_numpy(a_2).unsqueeze(0).repeat(240, 1, 1, 1)
    
    W_8, Sol_8=data_8['W'][:240,:,:,:], data_8['sol'][:240,:,:,:]
    X_8 = SPDE2D().FT_solver(W_8, O_T, O_X, O_Y)
    a_8 = SPDE2D().MC(X_8)
    a_8 = torch.from_numpy(a_8).unsqueeze(0).repeat(240, 1, 1, 1)

    W_32, Sol_32=data_32['W'][:240,:,:,:], data_32['sol'][:240,:,:,:]
    X_32 = SPDE2D().FT_solver(W_32, O_T, O_X, O_Y)
    a_32 = SPDE2D().MC(X_32)
    a_32 = torch.from_numpy(a_32).unsqueeze(0).repeat(240, 1, 1, 1)

    W_64, Sol_64=data_64['W'][:240,:,:,:], data_64['sol'][:240,:,:,:]
    X_64 = SPDE2D().FT_solver(W_64, O_T, O_X, O_Y)
    a_64 = SPDE2D().MC(X_64)
    a_64 = torch.from_numpy(a_64).unsqueeze(0).repeat(240, 1, 1, 1)

    W_128, Sol_128=data_128['W'][:240,:,:,:], data_128['sol'][:240,:,:,:]
    X_128 = SPDE2D().FT_solver(W_128, O_T, O_X, O_Y)
    a_128 = SPDE2D().MC(X_128)
    a_128 = torch.from_numpy(a_128).unsqueeze(0).repeat(240, 1, 1, 1)

    Sols = [Sol_2, Sol_8, Sol_32, Sol_64, Sol_128]
    Sol = np.concatenate(Sols, axis=0)

    Ws = [W_2, W_8, W_32, W_64, W_128]
    W = np.concatenate(Ws, axis=0)

    aepss = [a_2, a_8, a_32, a_64, a_128]
    aeps = torch.cat(aepss, dim=0)

    xi = torch.from_numpy(W.astype(np.float32))
    u = torch.from_numpy(Sol.astype(np.float32))

    n_samples = u.size(0)
    perm = torch.randperm(n_samples)
    shuffled_u = u[perm,...]
    shuffled_aeps = aeps[perm,...]
    shuffled_xi = xi[perm,...]
 
    u = shuffled_u[:1200,...]
    xi = shuffled_xi[:1200,...]
    aeps = shuffled_aeps[:1200,...]

    train_loader, test_loader = dataloader_nspde_2d(u=u, xi=xi, a_eps=aeps, ntrain=ntrain,
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


def hyperparameter_tuning(data_test_path, data_path_2, data_path_8, data_path_32, data_path_64, data_path_128, ntrain, nval, ntest, batch_size, epochs, learning_rate,
                 plateau_patience, plateau_terminate, print_every,
                 sub_x, T, sub_t,
                 hidden_channels, n_iter, modes1, modes2, solver,
                 log_file, checkpoint_file, final_checkpoint_file):
    # Load data
    data_2 = scipy.io.loadmat(data_path_2)
    data_8 = scipy.io.loadmat(data_path_8)
    data_32 = scipy.io.loadmat(data_path_32)
    data_64 = scipy.io.loadmat(data_path_64)
    data_128 = scipy.io.loadmat(data_path_128)
    data_test = scipy.io.loadmat(data_test_path)

    O_X, O_Y, O_T, W_2, Sol_2=data_2['X'].squeeze(), data_2['Y'].squeeze(), data_2['T'].squeeze(), data_2['W'][:240,:,:,:], data_2['sol'][:240,:,:,:]
    X_2 = SPDE2D().FT_solver(W_2, O_T, O_X, O_Y)
    a_2 = SPDE2D().MC(X_2)
    a_2 = torch.from_numpy(a_2).unsqueeze(0).repeat(240, 1, 1, 1)
    
    W_8, Sol_8=data_8['W'][:240,:,:,:], data_8['sol'][:240,:,:,:]
    X_8 = SPDE2D().FT_solver(W_8, O_T, O_X, O_Y)
    a_8 = SPDE2D().MC(X_8)
    a_8 = torch.from_numpy(a_8).unsqueeze(0).repeat(240, 1, 1, 1)

    W_32, Sol_32=data_32['W'][:240,:,:,:], data_32['sol'][:240,:,:,:]
    X_32 = SPDE2D().FT_solver(W_32, O_T, O_X, O_Y)
    a_32 = SPDE2D().MC(X_32)
    a_32 = torch.from_numpy(a_32).unsqueeze(0).repeat(240, 1, 1, 1)

    W_64, Sol_64=data_64['W'][:240,:,:,:], data_64['sol'][:240,:,:,:]
    X_64 = SPDE2D().FT_solver(W_64, O_T, O_X, O_Y)
    a_64 = SPDE2D().MC(X_64)
    a_64 = torch.from_numpy(a_64).unsqueeze(0).repeat(240, 1, 1, 1)

    W_128, Sol_128=data_128['W'][:240,:,:,:], data_128['sol'][:240,:,:,:]
    X_128 = SPDE2D().FT_solver(W_128, O_T, O_X, O_Y)
    a_128 = SPDE2D().MC(X_128)
    a_128 = torch.from_numpy(a_128).unsqueeze(0).repeat(240, 1, 1, 1)

    Sols = [Sol_2, Sol_8, Sol_32, Sol_64, Sol_128]
    Sol = np.concatenate(Sols, axis=0)

    Ws = [W_2, W_8, W_32, W_64, W_128]
    W = np.concatenate(Ws, axis=0)

    aepss = [a_2, a_8, a_32, a_64, a_128]
    aeps = torch.cat(aepss, dim=0)

    xi = torch.from_numpy(W.astype(np.float32))
    u = torch.from_numpy(Sol.astype(np.float32))

    W_test, Sol_test = data_test['W'], data_test['sol']
    X_test = SPDE2D().FT_solver(W_test, O_T, O_X, O_Y)
    a_test = SPDE2D().MC(X_test)
    a_test = torch.from_numpy(a_test).unsqueeze(0).repeat(1200, 1, 1, 1)
    
    xi_test = torch.from_numpy(W_test.astype(np.float32))
    data_test = torch.from_numpy(Sol_test.astype(np.float32))

    #shuffle
    n_samples = u.size(0)
    perm = torch.randperm(n_samples)
    shuffled_u = u[perm,...]
    shuffled_aeps = aeps[perm,...]
    shuffled_xi = xi[perm,...]
 
    u = shuffled_u[:1200,...]
    xi = shuffled_xi[:1200,...]
    aeps = shuffled_aeps[:1200,...]

    _, test_dl = dataloader_nspde_2d(u=data_test, xi=xi_test, a_eps=a_test, ntrain=ntrain+nval,
                                         ntest=ntest, T=T, sub_t=sub_t, sub_x=sub_x,
                                         batch_size=batch_size)

    train_dl, val_dl = dataloader_nspde_2d(u=u[:ntrain + nval], xi=xi[:ntrain + nval], a_eps=aeps[:ntrain + nval],
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