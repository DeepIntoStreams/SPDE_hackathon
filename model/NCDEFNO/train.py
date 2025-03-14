import torch
import torch.optim as optim
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from timeit import default_timer

from model.NCDEFNO.NCDEFNO_1D import NeuralCDE, dataloader_ncdeinf_1d, train_ncdeinf_1d
from model.nspde.utilities import *

import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = scipy.io.loadmat('/root/autodl-tmp/SPDE_hackathon/data_gen/examples/results/Phi41+/Phi41+_xi_{}.mat'.format(1200))

O_X, O_T, W, Sol = data['X'], data['T'], data['W'], data['sol']
xi = torch.from_numpy(W.astype(np.float32))
data = torch.from_numpy(Sol.astype(np.float32))

train_loader, test_loader = dataloader_ncdeinf_1d(data, xi, ntrain=1000, ntest=200, T=51,
                                                  sub_t=1, batch_size=20, dim_x=128,
                                                  interpolation='linear', dataset=None)

model = NeuralCDE(data_size=1, noise_size=1, hidden_channels=32, output_channels=1,
                  interpolation='linear').cuda()

print('The model has {} parameters'. format(count_params(model)))

loss = LpLoss(size_average=False)

model, losses_train, losses_test = train_ncdeinf_1d(model, train_loader, test_loader,
                                                    device, loss, batch_size=20, epochs=5000,
                                                    learning_rate=0.001, scheduler_step=100,
                                                    scheduler_gamma=0.5, print_every=1)

mem_log = []
for u0_, xi_, u_ in train_loader:
    input = u0_.to(device), xi_.to(device)
    break
try:
    mem_log.extend(log_mem(model, input, exp='baseline'))
except Exception as e:
    print(f'log_mem failed because of {e}')

df = pd.DataFrame(mem_log)
plot_mem(df, exps=['baseline'])
