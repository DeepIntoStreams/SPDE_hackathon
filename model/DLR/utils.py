# Adapted from https://github.com/sdogsq/DLR-Net

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from scipy import io
import h5py
from functools import reduce
from torch.utils.data import TensorDataset, DataLoader
import os
import os.path as osp
import sys
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(osp.join(current_directory, "..", ".."))

from model.DLR.Rule import Rule
from model.DLR.SPDEs import SPDE
from model.DLR.Graph import Graph
from model.DLR.RSlayer import ParabolicIntegrate, FNO_layer

def cacheXiFeature(graph, T, X, W, device, batch_size=100):
    '''
    return features only containing Xi
    '''
    InteLayer = ParabolicIntegrate(graph, T=T, X=X).to(device)
    WSet = TensorDataset(W)
    WLoader = DataLoader(WSet, batch_size=batch_size, shuffle=False)
    XiFeature = []
    for (W,) in WLoader:
        XiFeature.append(InteLayer(W=W.to(device)).to('cpu'))
    XiFeature = torch.cat(XiFeature, dim=0)
    return XiFeature

class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


def parabolic_graph(data, height):
    # create rule with additive width 3
    R = Rule(kernel_deg=2, noise_deg=-1.5, free_num=3)

    # initialize integration map I
    I = SPDE(BC='P', T=data['T'], X=data['X']).Integrate_Parabolic_trees

    G = Graph(integration=I, rule=R, height=height, deg=7.5)  # initialize graph

    extra_deg = 2
    key = "I_c[u_0]"

    graph = G.create_model_graph(data['W'][0],
                                 extra_planted={key: data['W'][0]},
                                 extra_deg={key: extra_deg})
    return graph


class rsnet(nn.Module):
    def __init__(self, graph, T, X):
        super().__init__()
        self.graph = graph
        self.F = len(graph) - 1
        self.FU0 = len([key for key in graph.keys() if 'u_0' in key])
        self.T = len(T)
        self.X = len(X)
        self.RSLayer0 = ParabolicIntegrate(graph, T=T, X=X)
        self.down0 = nn.Sequential(
            nn.Linear(self.F, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        # self.down1 = nn.Sequential(
        #     nn.Conv1d(self.T * self.F, 32 * self.T, kernel_size=1, groups = self.T),
        #     nn.GELU(),
        #     nn.Conv1d(32 * self.T, self.T, kernel_size = 1, groups = self.T)
        # )
        self.L = 4
        self.padding = 6
        modes1, modes2, width = 16, 16, 8  # 32, 24, 32
        self.net = [FNO_layer(modes1, modes2, width) for i in range(self.L - 1)]
        self.net += [FNO_layer(modes1, modes2, width, last=True)]
        self.net = nn.Sequential(*self.net)
        self.fc0 = nn.Linear(self.F + self.FU0 + 2, width)
        self.decoder = nn.Sequential(
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def get_grid(self, shape, device):
        batchsize, size_x, size_t = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).expand([batchsize, size_x, size_t, 1])
        gridt = torch.tensor(np.linspace(0, 1, size_t), dtype=torch.float)
        gridt = gridt.reshape(1, 1, size_t, 1).expand([batchsize, size_x, size_t, 1])
        return torch.cat((gridx, gridt), dim=-1).to(device)

    def forward(self, U0, W, Feature_Xi=None):
        '''
        U0: [B, N] initial condition
        W: [B, T, N] realizations of white noise
        Feature_Xi: [B, T, N, F] pre-computed features only containing Xi
        '''
        U0 = self.RSLayer0.I_c(U0)  # [B, T, N]

        R1 = self.RSLayer0(W=W, Latent=U0, XiFeature=Feature_Xi)  # [B, T, N, F + 1]

        O1 = R1[..., 1:]  # [B, T, N, F],  drop Xi
        U0 = self.down0(O1).squeeze(3)  # [B, T, N]
        R1 = self.RSLayer0(W=W, Latent=U0, XiFeature=Feature_Xi, returnU0Feature=True)

        R1 = torch.cat((O1, R1), dim=3)  # [B,T,N, F + FU0]
        grid = self.get_grid(R1.shape, R1.device)
        R1 = torch.cat((R1, grid), dim=-1)
        R1 = self.fc0(R1)
        R1 = R1.permute(0, 3, 2, 1)  # [B, Hidden, N, T]
        R1 = F.pad(R1, [0, self.padding])
        R1 = self.net(R1)
        R1 = R1[..., :-self.padding]
        R1 = R1.permute(0, 3, 2, 1)  # [B, T, N, Hidden]
        R1 = self.decoder(R1)  # [B, T, N, 1]
        return R1.squeeze()  # [B, T, N]


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (W, U0, F_Xi, Y) in enumerate(train_loader):
        W, U0, F_Xi, Y = W.to(device), U0.to(device), F_Xi.to(device), Y.to(device)
        optimizer.zero_grad()
        output = model(U0, W, F_Xi)
        loss = criterion(output[:, 1:, :], Y[:, 1:, :])
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader.dataset)


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (W, U0, F_Xi, Y) in enumerate(test_loader):
            W, U0, F_Xi, Y = W.to(device), U0.to(device), F_Xi.to(device), Y.to(device)
            output = model(U0, W, F_Xi)
            loss = criterion(output[:, 1:, :], Y[:, 1:, :])
            test_loss += loss.item()
    return test_loss / len(test_loader.dataset)
