#%cd Neural-SPDEs/
#!pip install -r requirements.txt

import torch
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from timeit import default_timer

import warnings
import torch.optim as optim
from utilities import *
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

""" The format of the figures. 
# To avoid type 3 figures. This may take a long time
!sudo apt install texlive texlive-latex-extra texlive-fonts-recommended dvipng
!pip install latex
!sudo apt install cm-super
# Using seaborn's style
plt.style.use('seaborn-colorblind')
width = 345
tex_fonts = {
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 24,
    "font.size": 24,
    "legend.fontsize": 24,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
}
plt.rcParams.update(tex_fonts)"""

# Load data set.
data = scipy.io.loadmat('E:/Doc/SignatureAndML/code/SPDE_hackathon/data_gen/results/Phi41+/Phi41+_xi_{}.mat'.format(1200))

# The data has been saved in the following format

# O_X[j] = x_j, 
# O_T[k] = t_k,
# W[i,j,k] = \xi^i(x_j,t_k), i=1,...,N_{train}+N_{test},
# Sol[i,j,k] = u^i(x_j,t_k), i=1,...,N_{train}+N_{test}.
O_X, O_T, W, Sol = data['X'], data['T'], data['W'], data['sol']
xi = torch.from_numpy(W.astype(np.float32))
data = torch.from_numpy(Sol.astype(np.float32))

from model.nspde.neural_spde import *

train_loader, test_loader = dataloader_nspde_1d(u=data, xi=xi, ntrain=1000, 
                                                ntest=200, T=51, sub_t=1, 
                                                batch_size=20, dim_x=128, 
                                                dataset='phi41')

# Define an NSPDE model
model = NeuralSPDE(dim=1, in_channels=1, noise_channels=1, hidden_channels=16, 
                   n_iter=4, modes1=32, modes2=32).cuda()
print('The model has {} parameters'. format(count_params(model)))

# Train the NSPDE model
loss = LpLoss(size_average=False)
model, losses_train, losses_test = train_nspde(model, train_loader, test_loader, 
                                                  device, loss, batch_size=20, epochs=5000, 
                                                  learning_rate=0.025, scheduler_step=100, 
                                                  scheduler_gamma=0.5, print_every=5)

torch.save(model.state_dict(), '../drive/MyDrive/data_phi41+/nspde_u0_xi_1200.pth')

plt.plot(np.arange(1,len(losses_train)*5, 5), losses_train, label='train')
plt.plot(np.arange(1,len(losses_test)*5, 5), losses_test, label='test')
plt.xlabel('Epoch')
plt.ylabel('Relative L2 loss')
plt.legend()
plt.show()

# Visualize pridictions
from utilities import plot_1d, contour_plot_1d
plot_1d(model, test_loader, device, i=1, T_=5, T=51, a=1)
contour_plot_1d(model, test_loader, device, O_X[0,:-1], O_T[0,:51])

# Memory profilling
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

# # Super-resolution demonstration
# # We start by loading a dataset with a finer spatial resolution
# batch_size = 20
# ntest = 200

# data = scipy.io.loadmat('../drive/MyDrive/data_phi41+/Phi41+_super_xi_u0_{}.mat'.format(ntest))
# O_X, O_T, W, Sol = data['X'], data['T'], data['W'], data['sol']
# xi = torch.from_numpy(W.astype(np.float32))
# data = torch.from_numpy(Sol.astype(np.float32))

# _, test_loader = dataloader_nspde_1d(u=data, xi=xi, ntrain=100, ntest=200, batch_size=batch_size, 
#                                      dim_x=512, dataset='phi41')

# # Then we load a model which has been trained on a lower resolution
# model = NeuralSPDE(dim=1, in_channels=1, noise_channels=1, hidden_channels=16, 
#                    n_iter=4, modes1=32, modes2=32).cuda()
# model.load_state_dict(torch.load('../drive/MyDrive/data_phi41+/nspde_u0_xi_1200_500epochs.pth'))

# # Finally we evaluate the model on the finer resolution dataset
# myloss = LpLoss(size_average=False)
# test_loss = 0.
# with torch.no_grad():
#     for u0_, xi_, u_ in test_loader:    
#         loss = 0.       
#         u0_, xi_, u_ = u0_.to(device), xi_.to(device), u_.to(device)
#         u_pred = model(u0_, xi_)
#         loss = myloss(u_pred[...,1:].reshape(batch_size, -1), u_[...,1:].reshape(batch_size, -1))
#         test_loss += loss.item()
# print('Super-resolution test Loss {:.6f} with {} spatial points'.format(test_loss / ntest, xi_.shape[-2]))