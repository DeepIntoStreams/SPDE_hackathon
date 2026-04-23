import scipy.io
import random
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import os.path as osp
import sys
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(osp.join(current_directory, "..",".."))
from model.NSPDE.utilities import *
from model.utilities import *
import warnings
import wandb
warnings.filterwarnings('ignore')

from evaluations import evaluate
from evaluations.metrics import (
    LpLossMetric,
    HsLossMetric,
    RMSEMetric,
    CovarianceMetric,
    AutoCorrelationMetric,
    CrossCorrelationMetric,
    MeanAbsDiffMetric,
    VARMetric,
    ESMetric,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint_file = '/root/autodl-tmp/SPDE_hackathon-phi43/model/result/NSPDE_Phi41+/Phi41+sigma1_xi_trc256_P_1200.pth'
data_path1 = '/root/autodl-tmp/SPDE_hackathon-phi43/data_gen/results/Phi41+/Phi41+sigma1_xi_trc256_P_1200.mat'
data_path2 = '/root/autodl-tmp/SPDE_hackathon-phi43/data_gen/results/Phi41+/Phi41+sigma1_xi_trc256_P_1200.mat'

# Load data
data1 = scipy.io.loadmat(data_path1)
data2 = scipy.io.loadmat(data_path2)
W1, Sol1 = data1['W'], data1['sol']
W2, Sol2 = data2['W'], data2['sol']
print('W1 shape:', W1.shape)
print('Sol1 shape:', Sol1.shape)
print('W2 shape:', W2.shape)
print('Sol2 shape:', Sol2.shape)
xi1 = torch.from_numpy(W1.astype(np.float32))
xi2 = torch.from_numpy(W2.astype(np.float32))
data1 = torch.from_numpy(Sol1.astype(np.float32))
data2 = torch.from_numpy(Sol2.astype(np.float32))

model = NeuralSPDE(dim=1, in_channels=1, noise_channels=1, hidden_channels=32,
                       n_iter=1, modes1=64, modes2=50).cuda()
print('The model has {} parameters'. format(count_params(model)))

model.load_state_dict(torch.load(checkpoint_file))

_, test_loader = dataloader_nspde_1d(u=data1, xi=xi1,
                                         ntrain=20,
                                         ntest=180,
                                         T=51,
                                         sub_t=1,
                                         batch_size=20,
                                         dim_x=128)


loss = 0
n_test = len(test_loader.dataset)
print(f"Number of test samples: {n_test}")
metric = LpLossMetric(size_average = False ,mode='rel')

for u0_, xi_, u_ in test_loader:
    u0_ = u0_.cuda()
    xi_ = xi_.cuda()
    u_ = u_.cuda()
    u_pred = model(u0_, xi_)
    scores = evaluate(u_, u_pred, [metric], batch_size=test_loader.batch_size)
    loss += scores['LpLossMetric'].item()

print(f"Average LpLoss over test set: {loss / n_test}")

