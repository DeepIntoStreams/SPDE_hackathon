import torch
import torch.optim as optim
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from timeit import default_timer

import warnings
warnings.filterwarnings('ignore')

from model.DeepONet.deepOnet import DeepONetCP, dataloader_deeponet_1d_xi, dataloader_deeponet_1d_u0, train_deepOnet_1d
from model.nspde.utilities import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data = scipy.io.loadmat('/root/autodl-tmp/SPDE_hackathon/data_gen/examples/results/Phi41+/Phi41+_xi_{}.mat'.format(1200))
data = scipy.io.loadmat('D:/MyPrograms/NSPDE/SPDE_hackathon-LzyCode/data_gen/examples/results/Phi41+/Phi41+_xi_1200.mat')

O_X, O_T, W, Sol = data['X'], data['T'], data['W'], data['sol']
xi = torch.from_numpy(W.astype(np.float32))
data = torch.from_numpy(Sol.astype(np.float32))

train_loader, test_loader, normalizer, grid = dataloader_deeponet_1d_xi(data, xi, ntrain=1000, ntest=200,
                                                            T=51, sub_t=1, batch_size=20,
                                                            dim_x=128, normalizer=False,
                                                            dataset=None)

model = DeepONetCP(branch_layer=[128] + [300, 200],
                    trunk_layer=[2] + [100, 200, 200]).to(device)

print('The model has {} parameters'. format(count_params(model)))

loss = LpLoss(size_average=False)

model, losses_train, losses_test = train_deepOnet_1d(model, train_loader, test_loader, grid,
                                                    normalizer, device, loss, batch_size=20,
                                                    epochs=500, learning_rate=0.001,
                                                    scheduler_step=100, scheduler_gamma=0.5,
                                                    print_every=1)

