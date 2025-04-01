import numpy as np
import pandas as pd
import math
from tqdm import tqdm
from time import time
import seaborn as sns
import torch


class Noise(object):

    def partition(self, a, b, dx):  # makes a partition of [a,b] of equal sizes dx
        return np.linspace(a, b, int((b - a) / dx) + 1)

    # Create l dimensional Brownian motion with time step = dt

    def BM(self, start, stop, dt, l):
        T = self.partition(start, stop, dt)
        # assign to each point of len(T) time point an N(0, \sqrt(dt)) standard l dimensional random variable
        BM = np.random.normal(scale=np.sqrt(dt), size=(len(T), l))
        BM[0] = 0  # set the initial value to 0
        BM = np.cumsum(BM, axis=0)  # cumulative sum: B_n = \sum_1^n N(0, \sqrt(dt))
        return BM

    # Create space time noise. White in time and with some correlation in space.
    # See Example 10.31 in "AN INTRODUCTION TO COMPUTATIONAL STOCHASTIC PDES" by Lord, Powell, Shardlow
    # X here is points in space as in SPDE1 function.
    def WN_space_time_single(self, s, t, dt, a, b, dx, correlation=None, numpy=True):

        # If correlation function is not given, use space-time white noise correlation fucntion
        if correlation is None:
            correlation = self.WN_corr

        T, X = self.partition(s, t, dt), self.partition(a, b, dx)  # time points, space points,
        N = len(X)
        # Create correlation Matrix in space
        space_corr = np.array([[correlation(x, j, dx * (N - 1)) for j in range(N)] for x in X])
        B = self.BM(s, t, dt, N)

        if numpy:
            return np.dot(B, space_corr.T)

        return pd.DataFrame(np.dot(B, space_corr), index=T, columns=X)

    def WN_space_time_many(self, s, t, dt, a, b, dx, num, correlation=None):

        return np.array([self.WN_space_time_single(s, t, dt, a, b, dx, correlation=correlation) for _ in range(num)])

    # Funciton for creating N random initial conditions of the form
    # \sum_{i = -p}^{i = p} a_k sin(k*\pi*x/scale)/ (1+|k|^decay) where a_k are i.i.d standard normal.
    def initial(self, N, X, p=10, decay=2, scaling=1, Dirichlet=False):
        scale = max(X) / scaling
        IC = []
        SIN = np.array(
            [[np.sin(k * np.pi * x / scale) / ((np.abs(k) + 1) ** decay) for k in range(-p, p + 1)] for x in X])
        for i in range(N):
            sins = np.random.normal(size=2 * p + 1)
            if Dirichlet:
                extra = 0
            else:
                extra = np.random.normal(size=1)
            IC.append(np.dot(SIN, sins) + extra)
            # enforcing periodic boundary condition without error
            IC[-1][-1] = IC[-1][0]

        return np.array(IC)

        # Correlation function that approximates WN in space.

    # See Example 10.31 in "AN INTRODUCTION TO COMPUTATIONAL STOCHASTIC PDES" by Lord, Powell, Shardlow
    def WN_corr(self, x, j, a):
        return np.sqrt(2 / a) * np.sin(j * np.pi * x / a)

    # save list of noises as a multilevel dataframe csv file
    def save(self, W, name):
        W.to_csv(name)

    def upload(self, name):
        Data = pd.read_csv(name, index_col=0, header=[0, 1])
        Data.columns = pd.MultiIndex.from_product([['W' + str(i + 1) for i in range(Data.columns.levshape[0])],
                                                   np.asarray(Data['W1'].columns, dtype=np.float16)])

        return Data


class GaussianRF(object):

    def __init__(self, dim, size, alpha=2, tau=3, sigma=None, boundary="periodic", device=None):

        self.dim = dim
        self.device = device

        if sigma is None:
            sigma = tau ** (0.5 * (2 * alpha - self.dim))

        k_max = size // 2

        if dim == 1:
            k = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                           torch.arange(start=-k_max, end=0, step=1, device=device)), 0)

            self.sqrt_eig = size * math.sqrt(2.0) * sigma * (
                        (4 * (math.pi ** 2) * (k ** 2) + tau ** 2) ** (-alpha / 2.0))
            self.sqrt_eig[0] = 0.0

        elif dim == 2:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size, 1)

            k_x = wavenumers.transpose(0, 1)
            k_y = wavenumers

            self.sqrt_eig = (size ** 2) * math.sqrt(2.0) * sigma * (
                        (4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2) + tau ** 2) ** (-alpha / 2.0))
            self.sqrt_eig[0, 0] = 0.0

        elif dim == 3:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size, size, 1)

            k_x = wavenumers.transpose(1, 2)
            k_y = wavenumers
            k_z = wavenumers.transpose(0, 2)

            self.sqrt_eig = (size ** 3) * math.sqrt(2.0) * sigma * (
                        (4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2 + k_z ** 2) + tau ** 2) ** (-alpha / 2.0))
            self.sqrt_eig[0, 0, 0] = 0.0

        self.size = []
        for j in range(self.dim):
            self.size.append(size)

        self.size = tuple(self.size)

    def sample(self, N):

        coeff = torch.randn(N, *self.size, 2, device=self.device)

        coeff[..., 0] = self.sqrt_eig * coeff[..., 0]
        coeff[..., 1] = self.sqrt_eig * coeff[..., 1]

        u = torch.fft.ifftn(torch.view_as_complex(coeff), dim=[1, 2]).real

        return u


def coloured_noise_2d(T, dt, X, N, num_samples, alpha=3, tau=3):
    GRF = GaussianRF(2, N, alpha=alpha, tau=tau)

    space = GRF.sample(num_samples)
    BM = Noise().BM(0, T, dt, num_samples)[:-1].T

    return np.array([np.multiply.outer(BM[i], space[i]) for i in range(num_samples)])


def get_twod_bj(dt, X, N, alpha):
    """
    Alg 10.5 Page 443 in the book "An Introduction to Computational Stochastic PDEs"
    """
    lambdax = 2 * np.pi * np.arange(- N // 2 + 1, N // 2 + 1) / X
    lambday = lambdax.copy()
    lambdaxx, lambdayy = np.meshgrid(lambdax, lambday)
    root_qj = np.exp(- alpha * (lambdaxx ** 2 + lambdayy ** 2) / 2)
    bj = root_qj * np.sqrt(dt) * N ** 2 / X
    return bj


def get_2d_dW(dt, X, N, alpha, num_samples):
    """
    Alg 10.6 Page 444 in the book "An Introduction to Computational Stochastic PDEs"
    """
    bj = get_twod_bj(dt, X, N, alpha)
    shape = (num_samples, N, N)
    nnr, nnc = np.random.standard_normal(size=shape), np.random.standard_normal(size=shape)
    nn = nnr + 1.j * nnc
    dW = np.fft.ifft2(bj[None, :, :] * nn, axes=[-2, -1])
    return dW.real


def get_W(T, dt, X, N, alpha, num_samples):
    M = int(np.ceil(T / dt))
    W = np.zeros([num_samples, M, N, N])
    for i in tqdm(range(1, M)):
        W[:, i, :, :] = get_2d_dW(dt, X, N, alpha, num_samples)
    return np.cumsum(W, axis=1)