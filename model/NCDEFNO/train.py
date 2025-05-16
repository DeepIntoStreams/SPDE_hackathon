import torch
import torch.optim as optim
import scipy.io
import hydra
from omegaconf import DictConfig
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
from model.NCDEFNO.NCDEFNO_1D import *
from model.utilities import *
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_training(data_path, ntrain, ntest, batch_size, epochs, learning_rate,
                 scheduler_step, scheduler_gamma, print_every,
                 dim_x, T, sub_t,
                 interpolation, dataset,
                 data_size, noise_size, hidden_channels, output_channels, solver, save_path):

    data = scipy.io.loadmat(data_path)

    O_X, O_T, W, Sol = data['X'], data['T'], data['W'], data['sol']
    xi = torch.from_numpy(W.astype(np.float32))
    data = torch.from_numpy(Sol.astype(np.float32))

    train_loader, test_loader = dataloader_ncdeinf_1d(data, xi, ntrain, ntest, T,
                                                      sub_t, batch_size, dim_x,
                                                      interpolation, dataset=dataset)

    model = NeuralCDE(data_size=data_size, noise_size=noise_size, hidden_channels=hidden_channels, output_channels=output_channels,
                      interpolation=interpolation, solver=solver).cuda()

    print('The model has {} parameters'. format(count_params(model)))

    loss = LpLoss(size_average=False)

    model, losses_train, losses_test = train_ncdeinf_1d(model, train_loader, test_loader,
                                                        device, loss, batch_size, epochs,
                                                        learning_rate, scheduler_step,
                                                        scheduler_gamma, print_every=print_every)

    torch.save(model.state_dict(), save_path)

    # mem_log = []
    # for u0_, xi_, u_ in train_loader:
    #     input = u0_.to(device), xi_.to(device)
    #     break
    # try:
    #     mem_log.extend(log_mem(model, input, exp='baseline'))
    # except Exception as e:
    #     print(f'log_mem failed because of {e}')
    #
    # df = pd.DataFrame(mem_log)
    # plot_mem(df, exps=['baseline'])

def hyperparameter_tuning(data_path, ntrain, nval, ntest, batch_size, epochs, learning_rate,
                 plateau_patience, plateau_terminate, print_every,
                 dim_x, T, sub_t,
                 interpolation, hidden_channels, solver,
                 log_file, checkpoint_file, final_checkpoint_file):

    data = scipy.io.loadmat(data_path)

    O_X, O_T, W, Sol = data['X'], data['T'], data['W'], data['sol']
    xi = torch.from_numpy(W.astype(np.float32))
    data = torch.from_numpy(Sol.astype(np.float32))

    _, test_dl = dataloader_ncdeinf_1d(u=data, xi=xi, ntrain=ntrain + nval,
                                       ntest=ntest, T=T, sub_t=sub_t,
                                       batch_size=batch_size, dim_x=dim_x, interpolation=interpolation)

    train_dl, val_dl = dataloader_ncdeinf_1d(u=data[:ntrain + nval], xi=xi[:ntrain + nval],
                                             ntrain=ntrain, ntest=nval, T=T, sub_t=sub_t,
                                             batch_size=batch_size, dim_x=dim_x, interpolation=interpolation)

    hyperparameter_search_ncdefno_1d(train_dl, val_dl, test_dl,
                                     d_h=hidden_channels, solver=solver, lr=learning_rate,
                                     epochs=epochs, print_every=print_every, plateau_patience=plateau_patience,
                                     plateau_terminate=plateau_terminate, log_file = log_file + '.csv',
                                     checkpoint_file=checkpoint_file,
                                     final_checkpoint_file=final_checkpoint_file)


@hydra.main(version_base=None, config_path="../config/", config_name="config_ncdefno_GL_xi")
def main(cfg: DictConfig):
    print('Training NCDE-FNO with wave xi ...')
    # run_training(**cfg.args)
    hyperparameter_tuning(**cfg.tuning)
    print('Done.')


if __name__ == '__main__':
    main()