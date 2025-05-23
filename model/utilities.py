# Adapted from https://github.com/crispitagorico/torchspde
# Modified for current implementation by the authors of SPDEBench

import torch
import operator
from functools import reduce
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
import os
import shutil

#loss function with rel/abs Lp loss
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

# Sobolev norm (HS norm)
# where we also compare the numerical derivatives between the output and target
class HsLoss(object):
    def __init__(self, d=2, p=2, k=1, a=None, group=False, size_average=True, reduction=True):
        super(HsLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.k = k
        self.balanced = group
        self.reduction = reduction
        self.size_average = size_average

        if a == None:
            a = [1,] * k
        self.a = a

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

    def __call__(self, x, y, a=None):
        nx = x.size()[1]
        ny = x.size()[2]
        k = self.k
        balanced = self.balanced
        a = self.a
        x = x.view(x.shape[0], nx, ny, -1)
        y = y.view(y.shape[0], nx, ny, -1)

        k_x = torch.cat((torch.arange(start=0, end=nx//2, step=1),torch.arange(start=-nx//2, end=0, step=1)), 0).reshape(nx,1).repeat(1,ny)
        k_y = torch.cat((torch.arange(start=0, end=ny//2, step=1),torch.arange(start=-ny//2, end=0, step=1)), 0).reshape(1,ny).repeat(nx,1)
        k_x = torch.abs(k_x).reshape(1,nx,ny,1).to(x.device)
        k_y = torch.abs(k_y).reshape(1,nx,ny,1).to(x.device)

        x = torch.fft.fftn(x, dim=[1, 2])
        y = torch.fft.fftn(y, dim=[1, 2])

        if balanced==False:
            weight = 1
            if k >= 1:
                weight += a[0]**2 * (k_x**2 + k_y**2)
            if k >= 2:
                weight += a[1]**2 * (k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
            weight = torch.sqrt(weight)
            loss = self.rel(x*weight, y*weight)
        else:
            loss = self.rel(x, y)
            if k >= 1:
                weight = a[0] * torch.sqrt(k_x**2 + k_y**2)
                loss += self.rel(x*weight, y*weight)
            if k >= 2:
                weight = a[1] * torch.sqrt(k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
                loss += self.rel(x*weight, y*weight)
            loss = loss / (k+1)

        return loss


# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c

# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path=None, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: None
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def plot_2d_xi(model, data_loader, device, i=1, T_=5, T=100, a=0, wb=None):
    # i: choose i_th data
    # T_: number of time points where you want to plot
    # T: total time steps
    # a: begin plot since time step = a

    for xi_, u_ in data_loader:  # choose the first batch to plot
        xi_ = xi_.to(device)
        u_ = u_.to(device)
        break

    with torch.no_grad():
        u_pred = model(xi_)
        u_pred = u_pred[..., 0]

    fig, ax = plt.subplots(2, T_, figsize=(T_*3, 6))

    vmin = min(u_[i].min(), u_pred[i].min()).detach().cpu().numpy()
    vmax = max(u_[i].max(), u_pred[i].max()).detach().cpu().numpy()
    im_norm = colors.Normalize(vmin=vmin, vmax=vmax)

    times = np.linspace(a, T-1, T_)
    for j in range(T_):
        t = int(times[j])

        # choose no.i data to plot
        true_slice = u_[i,...,t].detach().cpu().numpy()
        pred_slice = u_pred[i,...,t].detach().cpu().numpy()

        im1 = ax[0, j].imshow(true_slice, cmap='viridis', norm=im_norm)
        ax[0, j].set_title(f'True at time step {t}')

        im2 = ax[1, j].imshow(pred_slice, cmap='viridis', norm=im_norm)
        ax[1, j].set_title(f'Pred at time step {t}')

    # fig.colorbar(im1, fraction=.1)
    plt.subplots_adjust(bottom=0.05, right=0.9, top=0.95)
    cax = plt.axes((0.93, 0.1, 0.03, 0.8))  # [left, bottom, width, height]
    # plt.colorbar(cax=cax) # Wrong!

    fig.colorbar(im1, cax=cax, ax=ax, fraction=.1)

    if wb:
        wb.log({"True vs Pred on Test_data": wandb.Image(plt)})

    plt.show()


def save_model(cfg, sweep_id):

    print("Sweep completed. Searching for best model (min val_loss) ...")
    api = wandb.Api()
    sweep = api.sweep(f"{cfg.wandb.entity}/{cfg.wandb.project}/{sweep_id}")

    runs = sorted(sweep.runs, key=lambda run: run.summary.get("loss_val", float('inf')), reverse=False)
    if runs:
        best_run = runs[0]
        loss_val = best_run.summary.get("loss_val", float('inf'))
        loss_test = best_run.summary.get("loss_test", float('inf'))
        print(f"Best run: {best_run.name} | min Val Loss: {loss_val} | min Test Loss: {loss_test}")
        target_file = best_run.config.get("checkpoint_file", None)
        if target_file:
            save_dir = cfg.fixed.save.base_dir
            os.makedirs(save_dir, exist_ok=True)
            final_file = os.path.join(save_dir, f"{best_run.name}_sweep-{sweep_id}.pth")
            shutil.copy(target_file, final_file)
            print(f"Saved best run to {final_file}")
        else:
            print("No checkpoint file found in run config")
    else:
        print("No runs found in sweep")