# Adapted from https://github.com/crispitagorico/torchspde
# Modified for current implementation by the authors of SPDEBench

import os
import os.path as osp
import sys

current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(osp.join(current_directory, "..",".."))

import torch
import scipy.io
import h5py
import pandas as pd
import csv
import itertools
import matplotlib as mpl
from matplotlib.gridspec import SubplotSpec
from timeit import default_timer
from model.NSPDE.neural_spde import NeuralSPDE
from model.utilities import *

#===========================================================================
# Data Loaders for Neural SPDE
#===========================================================================

def dataloader_nspde_1d(u, xi=None, ntrain=1000, ntest=200, T=51, sub_t=1, batch_size=20, dim_x=128, dataset=None):

    if xi is None:
        print('There is no known forcing')

    if dataset=='phi41':
        T, sub_t = 51, 1
    elif dataset=='wave':
        T, sub_t = (u.shape[-1]+1)//2, 5

    u0_train = u[:ntrain, :dim_x, 0].unsqueeze(1)
    u_train = u[:ntrain, :dim_x, :T:sub_t]

    if xi is not None:
        xi_train = torch.diff(xi[:ntrain, :dim_x, 0:T:sub_t], dim=-1).unsqueeze(1)
        xi_train = torch.cat([torch.zeros_like(xi_train[..., 0].unsqueeze(-1)), xi_train], dim=-1)
    else:
        xi_train = torch.zeros_like(u_train).unsqueeze(1)

    u0_test = u[-ntest:, :dim_x, 0].unsqueeze(1)
    u_test = u[-ntest:, :dim_x, 0:T:sub_t]

    if xi is not None:
        xi_test = torch.diff(xi[-ntest:, :dim_x, 0:T:sub_t], dim=-1).unsqueeze(1)
        xi_test = torch.cat([torch.zeros_like(xi_test[..., 0].unsqueeze(-1)), xi_test], dim=-1)
    else:
        xi_test = torch.zeros_like(u_test).unsqueeze(1)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(u0_train, xi_train, u_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(u0_test, xi_test, u_test), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# def dataloader_nspde_diffeq_1d(u, xi=None, ntrain=1000, ntest=200, T=51, sub_t=1, batch_size=20, dim_x=128, dataset=None):

#     if xi is None:
#         print('There is no known forcing')

#     if dataset=='phi41':
#         T, sub_t = 51, 1
#     elif dataset=='wave':
#         T, sub_t = (u.shape[-1]+1)//2, 5

#     u0_train = u[:ntrain, :dim_x, 0].unsqueeze(1)
#     u_train = u[:ntrain, :dim_x, :T:sub_t]

#     if xi is not None:
#         xi_train = xi[:ntrain, :dim_x, 0:T:sub_t].unsqueeze(1)
#     else:
#         xi_train = torch.zeros_like(u_train)

#     u0_test = u[-ntest:, :dim_x, 0].unsqueeze(1)
#     u_test = u[-ntest:, :dim_x, 0:T:sub_t]

#     if xi is not None:
#         xi_test = xi[-ntest:, :dim_x, 0:T:sub_t].unsqueeze(1)
#     else:
#         xi_test = torch.zeros_like(u_test)

#     train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(u0_train, xi_train, u_train), batch_size=batch_size, shuffle=True)
#     test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(u0_test, xi_test, u_test), batch_size=batch_size, shuffle=False)

#     return train_loader, test_loader

# Get the data of Phi42


def get_data_Phi42(parquetfile, data_path):
    # Read Parquet file
    df = pd.read_parquet(parquetfile)

    # Extract dimension information
    T_values = df["T"].unique()
    X_values = df["X"].unique()
    Y_values = df["Y"].unique()

    len_T = len(T_values)       # e.g. 128
    len_X = len(X_values)       # e.g. 51
    len_Y = len(Y_values)       # e.g. 51

    # Extract data columns and convert to NumPy arrays
    W_flat = df["W"].to_numpy()
    sol_flat = df["sol"].to_numpy()

    # Reshape to original dimensions (1200, T, X, Y)
    W_restored = W_flat.reshape((1200, len_T, len_X, len_Y))
    sol_restored = sol_flat.reshape((1200, len_T, len_X, len_Y))

    # Extract scalar variables (assuming eps is stored as a constant column in Parquet)
    eps = df["eps"].iloc[0]  # Take first value (same for all rows)

    # Save as MAT file
    scipy.io.savemat(data_path + 'mat_data.mat', mdict={
        "W": W_restored,
        "sol": sol_restored,
        "eps": eps,
        # Coordinate information
        "X": X_values,
        "Y": Y_values,
        "T": T_values
    })
    return


# Get the data of KdV equations 
def get_data_KdV(parquetfile, data_path):
    # Read Parquet file
    df = pd.read_parquet(parquetfile)

    # Extract dimension information
    T_values = df["T"].unique()
    X_values = df["X"].unique()

    len_T = len(T_values)       # e.g. 128
    len_X = len(X_values)       # e.g. 51

    # Extract data columns and convert to NumPy arrays
    W_flat = df["W"].to_numpy()
    sol_flat = df["sol"].to_numpy()

    # Reshape to original dimensions (1200, T, X, Y)
    W_restored = W_flat.reshape((1200, len_X, len_T))
    sol_restored = sol_flat.reshape((1200, len_X, len_T))[:, 0:len_X-1, :]

    # Save as MAT file
    scipy.io.savemat(data_path + 'mat_data.mat', mdict={
        "W": W_restored,
        "sol": sol_restored,
        # Coordinate information
        "X": X_values,
        "T": T_values
    })
    return


# Get the data of 1d-equations except for KdV
def get_data_1d(parquetfile, data_path):
    # Read Parquet file
    df = pd.read_parquet(parquetfile)

    # Extract dimension information
    T_values = df["T"].unique()
    X_values = df["X"].unique()

    len_T = len(T_values)       # e.g. 128
    len_X = len(X_values)       # e.g. 51

    # Extract data columns and convert to NumPy arrays
    W_flat = df["W"].to_numpy()
    sol_flat = df["sol"].to_numpy()

    # Reshape to original dimensions (1200, T, X, Y)
    W_restored = W_flat.reshape((1200, len_X, len_T))
    sol_restored = sol_flat.reshape((1200, len_X, len_T))

    # Save as MAT file
    scipy.io.savemat(data_path + 'mat_data.mat', mdict={
        "W": W_restored,
        "sol": sol_restored,
        # Coordinate information
        "X": X_values,
        "T": T_values
    })
    return


# Get the data of NS equation
def get_data_NS(parquetfile, data_path):
    # Read Parquet file
    df = pd.read_parquet(parquetfile)

    # Extract dimension information
    T_values = df["T"].unique()

    len_T = len(T_values)       # e.g. 128

    # Extract data columns and convert to NumPy arrays
    W_flat = df["W"].to_numpy()
    sol_flat = df["sol"].to_numpy()

    # Reshape to original dimensions (1200, T, X, Y)
    W_restored = W_flat.reshape((1200, 64, 64, len_T))[:, :, :, 0:len_T-1]
    sol_restored = sol_flat.reshape((1200, 64, 64, len_T))

    # Save as MAT file
    scipy.io.savemat(data_path + 'mat_data.mat', mdict={
        "W": W_restored,
        "sol": sol_restored,
        # Coordinate information
        "T": T_values
    })
    return


def dataloader_nspde_2d(u, xi=None, ntrain=1000, ntest=200, T=51, sub_t=1, sub_x=4, batch_size=20, dataset=None):

    if xi is None:
        print('There is no known forcing')

    if dataset=='sns':
        T, sub_t, sub_x = 51, 1, 4

    u0_train = u[:ntrain, ::sub_x, ::sub_x, 0].unsqueeze(1)
    u_train = u[:ntrain, ::sub_x, ::sub_x, :T:sub_t]

    if xi is not None:
        xi_train = xi[:ntrain, ::sub_x, ::sub_x, 0:T:sub_t].unsqueeze(1)
    else:
        xi_train = torch.zeros_like(u_train).unsqueeze(1)

    u0_test = u[-ntest:, ::sub_x, ::sub_x, 0].unsqueeze(1)
    u_test = u[-ntest:, ::sub_x, ::sub_x, 0:T:sub_t]

    if xi is not None:
        xi_test = xi[-ntest:, ::sub_x, ::sub_x, 0:T:sub_t].unsqueeze(1)
    else:
        xi_test = torch.zeros_like(u_test).unsqueeze(1)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(u0_train, xi_train, u_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(u0_test, xi_test, u_test), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

#===========================================================================
# Training and Testing functionalities
#===========================================================================

def eval_nspde(model, test_dl, myloss, batch_size, device):

    ntest = len(test_dl.dataset)
    test_loss = 0.
    with torch.no_grad():
        for u0_, xi_, u_ in test_dl:    
            loss = 0.       
            u0_, xi_, u_ = u0_.to(device), xi_.to(device), u_.to(device)
            u_pred = model(u0_, xi_)
            loss = myloss(u_pred[...,1:].reshape(batch_size, -1), u_[...,1:].reshape(batch_size, -1))
            test_loss += loss.item()
    # print('Test Loss: {:.6f}'.format(test_loss / ntest))
    return test_loss / ntest

def train_nspde(model, train_loader, test_loader, device, myloss, batch_size=20, epochs=5000,
                learning_rate=0.001, scheduler_step=100, scheduler_gamma=0.5, print_every=20,
                weight_decay=1e-4, delta=0, factor=0.1,
                plateau_patience=None, plateau_terminate=None, time_train=False, time_eval=False,
                checkpoint_file='checkpoint.pt'):


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if plateau_patience is None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=plateau_patience, factor=factor, threshold=1e-6, min_lr=1e-7)
    if plateau_terminate is not None:
        early_stopping = EarlyStopping(patience=plateau_terminate, verbose=False, delta=delta, path=checkpoint_file)

    ntrain = len(train_loader.dataset)
    ntest = len(test_loader.dataset)

    losses_train = []
    losses_test = []

    times_train = [] 
    times_eval = []

    try:

        for ep in range(epochs):

            model.train()
            
            train_loss = 0.
            for u0_, xi_, u_ in train_loader:

                loss = 0.

                u0_ = u0_.to(device)
                xi_ = xi_.to(device)
                u_ = u_.to(device)

                t1 = default_timer()
                u_pred = model(u0_, xi_)
                loss = myloss(u_pred[..., 1:].reshape(batch_size, -1), u_[..., 1:].reshape(batch_size, -1))

                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                times_train.append(default_timer()-t1)

            test_loss = 0.
            with torch.no_grad():
                for u0_, xi_, u_ in test_loader:
                    
                    loss = 0.
                    
                    u0_ = u0_.to(device)
                    xi_ = xi_.to(device)
                    u_ = u_.to(device)

                    t1 = default_timer()

                    u_pred = model(u0_, xi_)

                    times_eval.append(default_timer()-t1)

                    loss = myloss(u_pred[..., 1:].reshape(batch_size, -1), u_[..., 1:].reshape(batch_size, -1))

                    test_loss += loss.item()
            
            if plateau_patience is None:
                scheduler.step()
            else:
                scheduler.step(test_loss/ntest)
            if plateau_terminate is not None:
                early_stopping(test_loss/ntest, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
        

            if ep % print_every == 0:
                losses_train.append(train_loss/ntrain)
                losses_test.append(test_loss/ntest)
                print('Epoch {:04d} | Total Train Loss {:.6f} | Total Val Loss {:.6f}'.format(ep, train_loss / ntrain, test_loss / ntest))

        if time_train and time_eval:
            return model, losses_train, losses_test, times_train, times_eval 
        elif time_train and not time_eval:
            return model, losses_train, losses_test, times_train
        elif time_eval and not time_train:
            return model, losses_train, losses_test, times_eval 
        else:
            return model, losses_train, losses_test
        
    except KeyboardInterrupt:
        if time_train and time_eval:
            return model, losses_train, losses_test, times_train, times_eval 
        elif time_train and not time_eval:
            return model, losses_train, losses_test, times_train
        elif time_eval and not time_train:
            return model, losses_train, losses_test, times_eval 
        else:
            return model, losses_train, losses_test


def hyperparameter_search_nspde_2d(train_dl, val_dl, test_dl, solver, d_h=[32], iter=[1, 2, 3], modes1=[32, 64], modes2=[32, 64],
                                epochs=500, print_every=20, lr=0.025, plateau_patience=100, plateau_terminate=100,
                                log_file='log_nspde', checkpoint_file='checkpoint.pt',
                                final_checkpoint_file='final.pt'):
    hyperparams = list(itertools.product(d_h, iter, modes1, modes2))

    loss = LpLoss(size_average=False)

    fieldnames = ['d_h', 'iter', 'modes1', 'modes2', 'nb_params', 'loss_train', 'loss_val', 'loss_test']
    with open(log_file, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)

    best_loss_val = 1000.

    for (_dh, _iter, _modes1, _modes2) in hyperparams:

        print('\n dh:{}, iter:{}, modes1:{}, modes2:{}'.format(_dh, _iter, _modes1, _modes2))

        model = NeuralSPDE(dim=2, in_channels=1, noise_channels=1, hidden_channels=_dh,
                           n_iter=_iter, modes1=_modes1, modes2=_modes2, solver=solver).cuda()

        nb_params = count_params(model)

        print('\n The model has {} parameters'.format(nb_params))

        # Train the model. The best model is checkpointed.
        _, _, _ = train_nspde(model, train_dl, val_dl, device, loss, batch_size=20, epochs=epochs, learning_rate=lr,
                              scheduler_step=500, scheduler_gamma=0.5, plateau_patience=plateau_patience,
                              plateau_terminate=plateau_terminate, print_every=print_every,
                              checkpoint_file=checkpoint_file)

        # load the best trained model
        model.load_state_dict(torch.load(checkpoint_file))

        # compute the test loss
        loss_test = eval_nspde(model, test_dl, loss, 20, device)

        # we also recompute the validation and train loss
        loss_train = eval_nspde(model, train_dl, loss, 20, device)
        loss_val = eval_nspde(model, val_dl, loss, 20, device)

        # if this configuration of hyperparameters is the best so far (determined wihtout using the test set), save it
        if loss_val < best_loss_val:
            torch.save(model.state_dict(), final_checkpoint_file)
            best_loss_val = loss_val

        # write results
        with open(log_file, 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([_dh, _iter, _modes1, _modes2, nb_params, loss_train, loss_val, loss_test])




#===============================================================================
# Utilities (adapted from https://github.com/zongyi-li/fourier_neural_operator)
#===============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# get grid
def get_grid(batch_size, dim_x, dim_y, dim_t=None):
    gridx = torch.linspace(0, 1, dim_x, dtype=torch.float)
    gridy = torch.linspace(0, 1, dim_y, dtype=torch.float)
    if dim_t:
        gridx = gridx.reshape(1, dim_x, 1, 1, 1).repeat([batch_size, 1, dim_y, dim_t, 1])
        gridy = gridy.reshape(1, 1, dim_y, 1, 1).repeat([batch_size, dim_x, 1, dim_t, 1])
        gridt = torch.linspace(0, 1, dim_t, dtype=torch.float)
        gridt = gridt.reshape(1, 1, 1, dim_t, 1).repeat([batch_size, dim_x, dim_y, 1, 1])
        return torch.cat((gridx, gridy, gridt), dim=-1).permute(0,4,1,2,3)
    gridx = gridx.reshape(1, dim_x, 1, 1).repeat([batch_size, 1, dim_y, 1])
    gridy = gridy.reshape(1, 1, dim_y, 1).repeat([batch_size, dim_x, 1, 1])
    return torch.cat((gridx, gridy), dim=-1).permute(0,3,1,2)

# reading data
class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float



def get_gpu_mem(synchronize=True, empty_cache=True):
    return torch.cuda.memory_allocated(), torch.cuda.memory_cached()


def generate_mem_hook(handle_ref, mem, idx, hook_type, exp):
    def hook(self, *args):
        if len(mem) == 0 or mem[-1]["exp"] != exp:
            call_idx = 0
        else:
            call_idx = mem[-1]["call_idx"] + 1

        mem_all, mem_cached = get_gpu_mem()
        torch.cuda.synchronize()
        mem.append({
            'layer_idx': idx,
            'call_idx': call_idx,
            'layer_type': type(self).__name__,
            'exp': exp,
            'hook_type': hook_type,
            'mem_all': mem_all,
            'mem_cached': mem_cached,
        })

    return hook


def add_memory_hooks(idx, mod, mem_log, exp, hr):
    h = mod.register_forward_pre_hook(generate_mem_hook(hr, mem_log, idx, 'pre', exp))
    hr.append(h)

    h = mod.register_forward_hook(generate_mem_hook(hr, mem_log, idx, 'fwd', exp))
    hr.append(h)

    h = mod.register_backward_hook(generate_mem_hook(hr, mem_log, idx, 'bwd', exp))
    hr.append(h)

def log_mem(model, inp, mem_log=None, exp=None, model_type='NSPDE'):
    mem_log = mem_log or []
    exp = exp or f'exp_{len(mem_log)}'
    hr = []
    for idx, module in enumerate(model.modules()):
        add_memory_hooks(idx, module, mem_log, exp, hr)
        
    try:
        if model_type in ['NSPDE', 'NCDE']:
            out = model(inp[0], inp[1])
        else:
            out = model(inp)
        
        loss = out.sum()
        loss.backward()
  
    finally:
        [h.remove() for h in hr]

        return mem_log


def plot_mem(df, exps=None, normalize_call_idx=True, normalize_mem_all=True, filter_fwd=False, return_df=False, output_file=None):
    if exps is None:
        exps = df.exp.drop_duplicates()

    fig, ax = plt.subplots(figsize=(20, 10))
    for exp in exps:
        df_ = df[df.exp == exp]

        if normalize_call_idx:
            df_.call_idx = df_.call_idx / df_.call_idx.max()

        if normalize_mem_all:
            df_.mem_all = df_.mem_all - df_[df_.call_idx == df_.call_idx.min()].mem_all.iloc[0]
            df_.mem_all = df_.mem_all // 2 ** 20

        if filter_fwd:
            layer_idx = 0
            callidx_stop = df_[(df_["layer_idx"] == layer_idx) & (df_["hook_type"] == "fwd")]["call_idx"].iloc[0]
            df_ = df_[df_["call_idx"] <= callidx_stop]
            # df_ = df_[df_.call_idx < df_[df_.layer_idx=='bwd'].call_idx.min()]

        plot = df_.plot(ax=ax, x='call_idx', y='mem_all', label=exp)
        print('Maximum memory: {} MB'.format(df_['mem_all'].max()))
        if output_file:
            plot.get_figure().savefig(output_file)

    if return_df:
        return df_



#===========================================================================
# The following utility returns the maximum memory usage
#===========================================================================

def get_memory(device, reset=False, in_mb=True):
    if device is None:
        return float('nan')
    if device.type == 'cuda':
        if reset:
            torch.cuda.reset_max_memory_allocated(device)
        bytes = torch.cuda.max_memory_allocated(device)
        if in_mb:
            bytes = bytes / 1024 / 1024
        return bytes
    else:
        return float('nan')
