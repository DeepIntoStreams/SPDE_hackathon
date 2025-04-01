import torch
import torch.optim as optim
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from timeit import default_timer
import signatory

from model.NRDE.NRDE import NeuralRDE, dataloader_nrde_1d, train_nrde_1d
from model.nspde.utilities import *

import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = scipy.io.loadmat('/root/autodl-tmp/SPDE_hackathon/data_gen/examples/results/Phi41+/Phi41+_xi_{}.mat'.format(1200))

O_X, O_T, W, Sol = data['X'], data['T'], data['W'], data['sol']
xi = torch.from_numpy(W.astype(np.float32))
data = torch.from_numpy(Sol.astype(np.float32))

dim_x = 128
train_load, test_load, I, noise_size, norm = dataloader_nrde_1d(u=data, xi=xi, ntrain=1000,
                                                                ntest=200, T=51, sub_t=1,
                                                                batch_size=20, dim_x=dim_x,
                                                                depth=2, window_length=10,
                                                                normalizer=True,
                                                                interpolation='linear',
                                                                dataset='phi41')

model = NeuralRDE(control_channels=noise_size, input_channels=dim_x,
                  hidden_channels=2, output_channels=dim_x, interval=I,
                  interpolation='linear').cuda()
print('The model has {} parameters'. format(count_params(model)))

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

loss = LpLoss(size_average=False)
model, losses_train, losses_test = train_nrde_1d(model, train_load, test_load, norm,
                                                device, loss, batch_size=20, epochs=5000,
                                                learning_rate=0.001, scheduler_step=100,
                                                scheduler_gamma=0.5, print_every=1)

