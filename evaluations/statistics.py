import torch
import numpy as np
from typing import Tuple


def to_numpy(x):
    """Convert a torch tensor to a numpy array."""
    return x.detach().cpu().numpy()


def cov_torch(x):
    """Estimates covariance matrix like numpy.cov"""
    device = x.device
    x = to_numpy(x)
    _, L, C = x.shape
    x = x.reshape(-1, L * C)
    return torch.from_numpy(np.cov(x, rowvar=False)).to(device).float()


def acf_torch(x: torch.Tensor, max_lag: int, dim: Tuple[int] = (0, 1)) -> torch.Tensor:
    """
    :param x: torch.Tensor [B, S, D]
    :param max_lag: int. specifies number of lags to compute the acf for
    :return: acf of x. [max_lag, D]
    """
    acf_list = list()
    x = x - x.mean((0, 1))
    std = torch.var(x, unbiased=False, dim=(0, 1))
    for i in range(max_lag):
        y = x[:, i:] * x[:, :-i] if i > 0 else torch.pow(x, 2)
        acf_i = torch.mean(y, dim) / std
        acf_list.append(acf_i)
    if dim == (0, 1):
        return torch.stack(acf_list)
    else:
        return torch.cat(acf_list, 1)


def non_stationary_acf_torch(X, symmetric=False):
    """
    Parameters
    ----------
    X : torch.Tensor, shape [B, T, D]
        Batch of time series.
    symmetric : bool
        If True, return full symmetric correlation matrix (including lower triangle and diagonal).
        If False, return only upper triangular part (s < t) with zeros elsewhere (diagonal and lower triangle set to 0).
    """
    B, T, D = X.shape
    correlations = torch.zeros(T, T, D, device=X.device, dtype=X.dtype)

    for d in range(D):
        # Extract time series for feature d across all samples: (B, T)
        x_d = X[:, :, d]
        # Compute correlation matrix (T, T) using torch.corrcoef
        if hasattr(torch, 'corrcoef'):
            corr_mat = torch.corrcoef(x_d.T)  # (T, T)
        else:
            corr_mat = torch.from_numpy(np.corrcoef(to_numpy(x_d).T)).to(X.device)
        
        if symmetric:
            # Full symmetric matrix (including diagonal and lower triangle)
            correlations[:, :, d] = corr_mat
        else:
            # Keep only upper triangular part (s < t), set diagonal and lower triangle to 0
            triu_mask = torch.triu(torch.ones(T, T, device=X.device), diagonal=1).bool()
            correlations[:, :, d] = corr_mat * triu_mask

    return correlations


def cacf_torch(x, lags: list, dim=(0, 1)):
    """
    Computes the cross-correlation between feature dimension and time dimension
    Parameters
    ----------
    x
    lags
    dim
    """
    # Define a helper function to get the lower triangular indices for a given dimension
    def get_lower_triangular_indices(n):
        return [list(x) for x in torch.tril_indices(n, n)]

    # Get the lower triangular indices for the input tensor x
    ind = get_lower_triangular_indices(x.shape[2])

    # Standardize the input tensor x along the given dimensions
    x = (x - x.mean(dim, keepdims=True)) / x.std(dim, keepdims=True)

    # Split the input tensor into left and right parts based on the lower triangular indices
    x_l = x[..., ind[0]]
    x_r = x[..., ind[1]]

    # Compute the cross-correlation at each lag and store in a list
    cacf_list = list()
    for i in range(lags):
        # Compute the element-wise product of the left and right parts, shifted by the lag if i > 0
        y = x_l[:, i:] * x_r[:, :-i] if i > 0 else x_l * x_r

        # Compute the mean of the product along the time dimension
        cacf_i = torch.mean(y, (1))

        # Append the result to the list of cross-correlations
        cacf_list.append(cacf_i)

    # Concatenate the cross-correlations across lags and reshape to the desired output shape
    cacf = torch.cat(cacf_list, 1)
    return cacf.reshape(cacf.shape[0], -1, len(ind[0]))


def non_stationary_cacf_torch(X):
    """
    Compute cross-correlation at lag 0 for each time step independently.
    Parameters
    ----------
    X (torch.Tensor): [B, T, D]
    """
    N, T, D = X.shape
    # Get indices for i < j
    i_idx, j_idx = torch.triu_indices(D, D, offset=1, device=X.device)
    M = len(i_idx)
    
    # Standardize and compute correlation per time step
    corr_ts = torch.zeros(T, M, device=X.device, dtype=X.dtype)
    for t in range(T):
        X_t = X[:, t, :]                     # (N, D)
        # Standardize across samples at this time step
        mean_t = X_t.mean(dim=0, keepdim=True)
        std_t = X_t.std(dim=0, keepdim=True)
        X_t_std = (X_t - mean_t) / std_t  # (N, D)
        # Correlation matrix (D, D)
        corr_mat = torch.corrcoef(X_t_std.T)  # (D, D)
        # Extract cross-correlations (i < j)
        corr_ts[t] = corr_mat[i_idx, j_idx]
    
    return corr_ts 


def rmse(x, y):
    return (x - y).pow(2).mean().sqrt()


def mean_abs_diff(den1: torch.Tensor, den2: torch.Tensor):
    return torch.mean(torch.abs(den1 - den2), 0)
