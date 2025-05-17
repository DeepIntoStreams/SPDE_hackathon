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

from model.DLR.RSlayer_2d import ParabolicIntegrate_2d, FNO_layer

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

def cacheXiFeature_2d(graph, T, X, Y, W, eps, device, batch_size = 100):
    '''
    return features only containing Xi
    '''
    InteLayer = ParabolicIntegrate_2d(graph, T = T, X = X, Y = Y, eps = eps).to(device)
    WSet = TensorDataset(W)
    WLoader = DataLoader(WSet, batch_size=batch_size, shuffle=False)
    XiFeature = []
    for (W, ) in WLoader:
        XiFeature.append(InteLayer(W = W.to(device)).to('cpu'))
    XiFeature = torch.cat(XiFeature, dim = 0)
    return XiFeature

def mat2data(reader, sub_t, sub_x):
    data = {}
    data['T'] = reader.read_field('t').squeeze()[:1000:10 * sub_t].squeeze()  # if data['t'] was not downsampled before
    # data['T'] = reader.read_field('t').squeeze()
    data['Solution'] = reader.read_field('sol')
    data['W'] = reader.read_field('forcing')
    spoints = np.linspace(0, 1, data['W'].shape[1] // sub_x)
    data['Y'], data['X'] = np.meshgrid(spoints, spoints)
    return data

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
            self.data = io.loadmat(self.file_path)
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

def dataloader_2d(u, xi=None, ntrain=1000, ntest=200, T=51, sub_t=1, sub_x=4):
    if xi is None:
        print('There is no known forcing')

    u0_train = u[:ntrain, ::sub_x, ::sub_x, 0]  # .unsqueeze(1)
    u_train = u[:ntrain, ::sub_x, ::sub_x, :T:sub_t]

    if xi is not None:
        xi_train = xi[:ntrain, ::sub_x, ::sub_x, 0:T:sub_t]  # .unsqueeze(1)
    else:
        xi_train = torch.zeros_like(u_train)

    u0_test = u[-ntest:, ::sub_x, ::sub_x, 0]  # .unsqueeze(1)
    u_test = u[-ntest:, ::sub_x, ::sub_x, 0:T:sub_t]

    if xi is not None:
        xi_test = xi[-ntest:, ::sub_x, ::sub_x, 0:T:sub_t]  # .unsqueeze(1)
    else:
        xi_test = torch.zeros_like(u_test)

    return (xi_train.transpose(0, 3, 1, 2), xi_test.transpose(0, 3, 1, 2),
            u0_train, u0_test,
            u_train.transpose(0, 3, 1, 2), u_test.transpose(0, 3, 1, 2))


def NS_graph(data, height):
    # create rule with additive width 2
    R = Rule(kernel_deg=2, noise_deg=-2, free_num=2)

    # initialize integration map I
    I = SPDE(BC='P', T=data['T'], X=data['X']).Integrate_Parabolic_trees_2d

    G = Graph(integration=I, rule=R, height=height, deg=7.5, derivative=True)  # initialize graph

    extra_deg = 2
    key = "I_c[u_0]"
    SZ = data['X'].shape
    graph = G.create_model_graph_2d(np.zeros((len(data['T']), *SZ)), data['X'],
                                    extra_planted={key: np.zeros((len(data['T']), *SZ))},
                                    extra_deg={key: extra_deg})
    # delete unused derivative features
    used = set().union(*[{IZ for IZ in graph[key].keys()} for key in graph.keys() if key[:2] == 'I['])
    graph = {IZ: graph[IZ] for IZ in graph if IZ[:2] == 'I[' or IZ in used}
    if (key not in graph.keys()):
        graph = list(graph.items())
        graph.insert(1, (key, dict()))
        graph = dict(graph)
    return graph


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (W, U0, F_Xi, Y) in enumerate(train_loader):
        W, U0, F_Xi, Y = W.to(device), U0.to(device), F_Xi.to(device), Y.to(device)
        optimizer.zero_grad()
        output = model(U0, W, F_Xi)
        loss = criterion(output[:, 1:, ...], Y[:, 1:, ...])
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader.dataset)


def saveplot(pred, u, epoch):
    import matplotlib.pyplot as plt
    T = pred.shape[1]
    fig, ax = plt.subplots(2, T, figsize=(2 * T, 4))
    for i in range(T):
        if torch.is_tensor(u):
            ax[0][i].contourf(u[0, i, ...].detach().cpu().numpy())
            ax[1][i].contourf(pred[0, i, ...].detach().cpu().numpy())
        else:
            ax[0][i].contourf(u[0, i, ...])
            ax[1][i].contourf(pred[0, i, ...])
        ax[0][i].set_title(f't = {i}')
    plt.savefig(f"./fig/{epoch}.pdf", bbox_inches='tight')
    plt.clf()


def test(model, device, test_loader, criterion, epoch=None):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (W, U0, F_Xi, Y) in enumerate(test_loader):
            W, U0, F_Xi, Y = W.to(device), U0.to(device), F_Xi.to(device), Y.to(device)
            # print('batch_idx', batch_idx, ' | W[0,:5,1,1]:', W[0,:5,8,8])
            # print('batch_idx', batch_idx, ' | U0[0,1,1]:', U0[0,8,8])
            # print('batch_idx', batch_idx, ' | F_Xi[1,:5,1,1]:', F_Xi[1,:5,8,8,:5])
            # print('batch_idx', batch_idx, ' | Y[1,:5,1,1]:', Y[1,:5,8,8])
            output = model(U0, W, F_Xi)
            # print('batch_idx', batch_idx, ' | output[1,:5,1,1]:', output[1,:5,8,8])
            loss = criterion(output[:, 1:, ...], Y[:, 1:, ...])
            # print('loss', loss)
            test_loss += loss.item()
        # saveplot(output, Y, epoch)
    return test_loss / len(test_loader.dataset)


class rsnet_2d(nn.Module):
    def __init__(self, graph, T, X, Y, nu):
        super().__init__()
        self.graph = graph
        self.vkeys = [key for key in graph.keys() if key[-1] is ']']
        self.F = len(self.vkeys)
        self.FU0 = len([key for key in self.vkeys if 'u_0' in key])
        self.T = len(T)
        self.X = len(X)
        self.Y = len(Y)
        self.RSLayer0 = ParabolicIntegrate_2d(graph, T=T, X=X, Y=Y, eps=nu)
        self.down0 = nn.Sequential(
            nn.Linear(1 + self.F, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )

        self.L = 4
        self.padding = 6
        modes1, modes2, modes3, width = 8, 8, 8, 8  # 16, 16, 10, 8 #8, 8, 8, 20
        self.net = [FNO_layer(modes1, modes2, modes3, width) for i in range(self.L - 1)]
        self.net += [FNO_layer(modes1, modes2, modes3, width, last=True)]
        self.net = nn.Sequential(*self.net)
        self.fc0 = nn.Linear(1 + self.F + self.FU0 + 3, width)
        self.decoder = nn.Sequential(
            nn.Linear(width, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )
        grid = self.get_grid(self.T, self.X, self.Y)
        self.register_buffer("grid", grid)

    def get_grid(self, T, X, Y):
        batchsize, size_x, size_y, size_z = 1, T, X, Y
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1)

    def forward(self, U0, W, Feature_Xi=None):
        '''
        U0: [B, X, Y] initial condition
        W: [B, T, X, Y] realizations of white noise
        Feature_Xi: [B, T, X, Y, F] pre-computed features only containing Xi
        '''
        U0 = self.RSLayer0.I_c(U0)  # [B, T, X, Y]
        R1 = self.RSLayer0(W=W, Latent=U0, XiFeature=Feature_Xi, returnFeature='normal')
        O1 = R1  # [B, T, X, Y, F + 1] with xi
        U0 = self.down0(O1).squeeze()  # [B, T, X, Y]
        R1 = self.RSLayer0(W=W, Latent=U0, XiFeature=Feature_Xi, returnFeature='U0')
        R1 = torch.cat((R1, O1, self.grid.expand(R1.shape[0], -1, -1, -1, -1)), dim=-1)  # [B, T, X, Y, 1 + F + FU0 + 3]
        R1 = self.fc0(R1)
        R1 = R1.permute(0, 4, 2, 3, 1)  # [B, Hidden, X, Y, T]
        R1 = F.pad(R1, [0, self.padding])
        R1 = self.net(R1)
        R1 = R1[..., :-self.padding]
        R1 = R1.permute(0, 4, 2, 3, 1)  # [B, T, X, Y, Hidden]
        R1 = self.decoder(R1)  # [B, T, X, Y, 1]
        return R1.squeeze()  # [B, T, X, Y]

