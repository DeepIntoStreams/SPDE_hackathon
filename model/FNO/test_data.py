import torch
import torch.optim as optim
import scipy.io
import wandb
import random
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from timeit import default_timer
import shutil
import os
import os.path as osp
import sys
from FNO2D import *
from model.utilities import *
data = scipy.io.loadmat('/root/autodl-tmp/SPDE_hackathon/data_gen/NS_xi_nu4/merged_ns_xi.mat')
W, Sol = data['forcing'], data['sol']
print('data shape:')
print(W.shape)
print(Sol.shape)

xi = torch.from_numpy(W.astype(np.float32))
data = torch.from_numpy(Sol.astype(np.float32))
ntrain = 3500
nval = 750
ntest = 750
sub_x=1
T=100
sub_t=2
batch_size=20
print('1 begin dl split')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_, test_loader = dataloader_fno_2d_xi(u=data, xi=xi,
                                                 ntrain=ntrain+nval,
                                                 ntest=ntest,
                                                 T=T,
                                                 sub_t=sub_t,
                                                 sub_x=sub_x,
                                                 batch_size=batch_size)
train_loader, val_loader = dataloader_fno_2d_xi(u=data[:ntrain + nval], xi=xi[:ntrain + nval],
                                      ntrain=ntrain,
                                      ntest=nval,
                                      T=T,
                                      sub_t=sub_t,
                                      sub_x=sub_x,
                                      batch_size=batch_size)
plot_2d(train_loader, val_loader, device, i=1, j=2, T_=10, T=T//sub_t)