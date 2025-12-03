import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_wavelets import DWTForward, DWTInverse

from model.utilities import *


class WaveConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, level, dummy_input):
        super(WaveConv2d, self).__init__()
        """
        2D Wavelet Layer. 
        Perform DWT -> Linear Transform on Coeffs -> IDWT
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level

        self.dwt = DWTForward(J=self.level, mode='symmetric', wave='db6').to(dummy_input.device)
        self.idwt = DWTInverse(mode='symmetric', wave='db6').to(dummy_input.device)

        # Get coefficient shapes via a dummy run for shape-adaptive weights
        yl, yh = self.dwt(dummy_input)
        self.yl_shape = yl.shape[-2:]
        self.yh_shape = yh[-1].shape[-2:]

        self.scale = (1 / (in_channels * out_channels))

        # 1. Weights for the low-frequency part
        self.w_yl = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, *self.yl_shape))

        # 2. Weights for the high-frequency detail part
        self.w_yh = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, 3, *self.yh_shape))

    def mul2d(self, input, weights):
        # (batch, in_channel, ...), (in_channel, out_channel, ...) -> (batch, out_channel, ...)
         return torch.einsum("bixy,ioxy->boxy", input, weights)

    def mul2d_detail(self, input, weights):
        # (B, C, 3, H, W)
        return torch.einsum("bicxy,iocxy->bocxy", input, weights)

    def forward(self, x):
        yl, yh = self.dwt(x)

        yl_out = self.mul2d(yl, self.w_yl)

        yh_out = yh[:-1] + [self.mul2d_detail(yh[-1], self.w_yh)]

        x = self.idwt((yl_out, yh_out))
        return x


class WNO_layer(nn.Module):
    def __init__(self, width, level, dummy_input, last=False):
        super(WNO_layer, self).__init__()
        self.last = last

        # dummy_input is passed to initialize the shape of WaveConv2d weights
        self.conv = WaveConv2d(width, width, level, dummy_input)
        self.w = nn.Conv2d(width, width, 1)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.w(x)
        if x1.shape[-1] != x2.shape[-1] or x1.shape[-2] != x2.shape[-2]:
            x1 = x1[..., :x2.shape[-2], :x2.shape[-1]]
        x = x1 + x2
        if not self.last:
            x = F.gelu(x)
        return x


class WNO_space1D_time(nn.Module):
    def __init__(self, level, width, L, T, input_sample):
        super(WNO_space1D_time, self).__init__()

        """
        Input: (batch, dim_x, T_out, T_in) 
        """
        self.level = level
        self.width = width
        self.L = L
        self.padding = 6

        # Input channel dimension is T_in + 2 (T_in + spatial grid + temporal grid)
        self.fc0 = nn.Linear(T + 2, self.width)

        # Prepare a dummy input for layer initialization
        # Shape after fc0 and permute: (1, width, dim_x, T_out + pad)
        dummy_tensor = torch.zeros(1, width, input_sample.shape[1], input_sample.shape[2] + self.padding).to(
            input_sample.device)

        layers = []
        for i in range(self.L - 1):
            layers.append(WNO_layer(width, level, dummy_tensor))
        layers.append(WNO_layer(width, level, dummy_tensor, last=True))

        self.net = nn.Sequential(*layers)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # x shape: (batch, dim_x, T_out, T)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)  # -> (B, X, T_out, T+2)

        x = self.fc0(x)  # -> (B, X, T_out, width)
        x = x.permute(0, 3, 1, 2)  # -> (B, width, X, T_out)

        # Padding
        x = F.pad(x, [0, self.padding])  # -> (B, width, X, T_out + pad)

        # WNO Layers
        x = self.net(x)

        # Remove Padding
        x = x[..., :-self.padding]

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_t = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_t, 1])
        gridt = torch.tensor(np.linspace(0, 1, size_t), dtype=torch.float)
        gridt = gridt.reshape(1, 1, size_t, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridt), dim=-1).to(device)



def dataloader_wno_1d_xi(u, xi, ntrain=1000, ntest=200, T=51, sub_t=1, batch_size=20, dim_x=128):

    u_train = u[:ntrain, :dim_x, 0:T:sub_t]
    xi_ = torch.diff(xi[:ntrain, :dim_x, 0:T:sub_t], dim=-1)
    xi_ = torch.cat([torch.zeros_like(xi_[..., 0].unsqueeze(-1)), xi_], dim=-1)
    xi_train = xi_[:ntrain].reshape(ntrain, dim_x, 1, xi_.shape[-1]).repeat([1, 1, xi_.shape[-1], 1])

    u_test = u[-ntest:, :dim_x, 0:T:sub_t]
    xi_ = torch.diff(xi[-ntest:, :dim_x, 0:T:sub_t], dim=-1)
    xi_ = torch.cat([torch.zeros_like(xi_[..., 0].unsqueeze(-1)), xi_], dim=-1)
    xi_test = xi_[-ntest:].reshape(ntest, dim_x, 1, xi_.shape[-1]).repeat([1, 1, xi_.shape[-1], 1])

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xi_train, u_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xi_test, u_test), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader



def eval_wno_1d(model, test_dl, myloss, batch_size, device):

    ntest = len(test_dl.dataset)
    test_loss = 0.
    with torch.no_grad():
        for xi_, u_ in test_dl:
            loss = 0.
            xi_, u_ = xi_.to(device), u_.to(device)
            u_pred = model(xi_)
            u_pred = u_pred[..., 0]
            loss = myloss(u_pred[..., 1:].reshape(batch_size, -1), u_[..., 1:].reshape(batch_size, -1))
            test_loss += loss.item()
    # print('Test Loss: {:.6f}'.format(test_loss / ntest))
    return test_loss / ntest


def train_wno_1d(model, train_loader, test_loader, device, myloss, batch_size=20, epochs=5000, learning_rate=0.001,
                 weight_decay=1e-4, scheduler_step=100, scheduler_gamma=0.5, plateau_patience=None, delta=0,
                 plateau_terminate=None, print_every=20, checkpoint_file='checkpoint.pt'):


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if plateau_patience is None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=plateau_patience, threshold=1e-6, min_lr=1e-7)
    if plateau_terminate is not None:
        early_stopping = EarlyStopping(patience=plateau_terminate, verbose=False, delta=delta, path=checkpoint_file)

    ntrain = len(train_loader.dataset)
    ntest = len(test_loader.dataset)

    losses_train = []
    losses_test = []

    try:

        for ep in range(epochs):

            model.train()

            train_loss = 0.
            for xi_, u_ in train_loader:

                loss = 0.
                xi_ = xi_.to(device)
                u_ = u_.to(device)

                u_pred = model(xi_)
                u_pred = u_pred[..., 0]
                loss = myloss(u_pred[..., 1:].reshape(batch_size, -1), u_[..., 1:].reshape(batch_size, -1))

                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            test_loss = 0.
            with torch.no_grad():
                for xi_, u_ in test_loader:

                    loss = 0.

                    xi_ = xi_.to(device)
                    u_ = u_.to(device)

                    u_pred = model(xi_)
                    u_pred = u_pred[..., 0]
                    loss = myloss(u_pred[..., 1:].reshape(batch_size, -1), u_[..., 1:].reshape(batch_size, -1))

                    test_loss += loss.item()

            if plateau_patience is None:
                scheduler.step()
            else:
                scheduler.step(test_loss/ntest)
            if plateau_terminate is not None:
                early_stopping(test_loss/ntest, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            if ep % print_every == 0:
                losses_train.append(train_loss/ntrain)
                losses_test.append(test_loss/ntest)
                print('Epoch {:04d} | Total Train Loss {:.6f} | Total Test Loss {:.6f}'.format(ep, train_loss / ntrain, test_loss / ntest))

        return model, losses_train, losses_test

    except KeyboardInterrupt:

        return model, losses_train, losses_test



