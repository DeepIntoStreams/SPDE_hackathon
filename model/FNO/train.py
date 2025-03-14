import torch
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from timeit import default_timer
from FNO1D import FNO_space1D_time, dataloader_fno_1d_xi, dataloader_fno_1d_u0, train_fno_1d

import warnings
import torch.optim as optim
from model.nspde.utilities import *
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data set.
data = scipy.io.loadmat('/root/autodl-tmp/data_gen/examples/results/Phi41+/Phi41+_xi_{}.mat'.format(1200))

O_X, O_T, W, Sol = data['X'], data['T'], data['W'], data['sol']
xi = torch.from_numpy(W.astype(np.float32))
data = torch.from_numpy(Sol.astype(np.float32))

train_loader, test_loader = dataloader_fno_1d_xi(u=data, xi=xi, ntrain=1000,
                                                 ntest=200, T=51, sub_t=1, batch_size=20,
                                                 dim_x=128, dataset='phi41')

model = FNO_space1D_time(modes1=32, modes2=24, width=32, T=1, L=4).cuda()

print('The model has {} parameters'. format(count_params(model)))

loss = LpLoss(size_average=False)

model, losses_train, losses_test = train_fno_1d(model, train_loader, test_loader,
                                                device, loss, batch_size=20, epochs=5000,
                                                learning_rate=0.001, scheduler_step=100,
                                                scheduler_gamma=0.5, print_every=5)

torch.save(model.state_dict(), '/root/autodl-tmp/data_gen/examples/results/Phi41+/fno_xi_1200.pth')


