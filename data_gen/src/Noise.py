import numpy as np
import pandas as pd
import math
from tqdm import tqdm
import torch

def partition(a, b, dx):  # makes a partition of [a,b] of equal sizes dx
    return np.linspace(a, b, int((b - a) / dx) + 1)

class Noise():

    # Create l dimensional Brownian motion with time step = dt

    def BM(self, start, stop, dt, l):
        T = partition(start, stop, dt)
        # assign to each point of len(T) time point an N(0, \sqrt(dt)) standard l dimensional random variable
        BM = np.random.normal(scale=np.sqrt(dt), size=(len(T), l))
        BM[0] = 0  # set the initial value to 0
        BM = np.cumsum(BM, axis=0)  # cumulative sum: B_n = \sum_1^n N(0, \sqrt(dt))
        return BM

    # Create space-time noise. White in time and with some correlation in space.
    # See Example 10.31 in "AN INTRODUCTION TO COMPUTATIONAL STOCHASTIC PDES" by Lord, Powell, Shardlow
    # X here is points in space as in SPDE1 function.
    def WN_space_time_single(self, s, t, dt, a, b, dx, J=None, correlation=None, numpy=True):

        # If correlation function is not given, use space-time white noise correlation fucntion
        if correlation is None:
            correlation = self.WN_corr

        T, X = partition(s, t, dt), partition(a, b, dx)  # time points, space points,
        N = len(X)
        if J == None:
            J = 2

        # Create correlation Matrix in space
        space_corr = np.array([[correlation(x, j, dx * (N - 1)) for j in range(J+1)] for x in X])
        B = self.BM(s, t, dt, J+1)

        if numpy:
            return np.dot(B, space_corr.T)

        return pd.DataFrame(np.dot(B, space_corr), index=T, columns=X)

    def WN_space_time_many(self, s, t, dt, a, b, dx, num, J=None, correlation=None):
        return np.array([self.WN_space_time_single(s, t, dt, a, b, dx, J, correlation=correlation) for _ in range(num)])

    # Funciton for creating N random initial conditions of the form
    # \sum_{i = -p}^{i = p} a_k sin(k*\pi*x/scale)/ (1+|k|^decay) where a_k are i.i.d standard normal.
    def initial(self, N, X, p=10, decay=2, scaling=1, Dirichlet=False):
        scale = max(X) / scaling  # for example, here max(X)=0.5, then scale=0.5 when scaling=1
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


class Noise2D(object):
    def partition(self, a, b, dx):  # makes a partition of [a,b] of equal sizes dx
        return np.linspace(a, b, int((b - a) / dx) + 1)

    def partition_2d(self, a, b, dx, c, d, dy):  # makes a partition of [a,b]×[c,d] of equal sizes dx, dy
        X = np.linspace(a, b, int((b - a) / dx) + 1)
        Y = np.linspace(c, d, int((d - c) / dy) + 1)
        # xx, yy = np.meshgrid(X, Y, indexing='ij')
        return X, Y

    # Create 1 dimensional Brownian motion with time step = dt

    def BM(self, start, stop, dt, lx, ly):
        T = self.partition(start, stop, dt)
        # assign to each point of len(T) time point an N(0, \sqrt(dt)) standard l dimensional random variable
        BM = np.random.normal(scale=np.sqrt(dt), size=(len(T), lx * ly))
        BM[0] = 0  # set the initial value to 0
        BM = np.cumsum(BM, axis=0)  # cumulative sum: B_n = \sum_1^n N(0, \sqrt(dt))
        return BM

    # Create space time noise. White in time and with some correlation in space.
    def WN_space_time_2d_single(self, s, t, dt, a, b, dx, c, d, dy, Jx=None, Ky=None, correlation=None, numpy=True):
        """
        Parameters:
            s, t: time interval [s, t]
            dt: time step size
            a, b: x-dimension spatial interval [a, b]
            dx: x-dimension spatial step size
            c, d: y-dimension spatial interval [c, d]
            dy: y-dimension spatial step size
            correlation: spatial correlation function (defaults to the 2D sinusoidal basis)
            Jx, Ky: Cut down of the Cylindrical Wiener process' series 
        """
        # If correlation function is not given, use space-time white noise correlation fucntion
        if correlation is None:
            correlation = self.WN_corr_2d

        # space points
        X, Y = self.partition_2d(a, b, dx, c, d, dy)  
        Nx = len(X)
        Ny = len(Y) 
        # time points
        T = self.partition(s, t, dt)

        # Cut down of the Cylindrical Wiener process' series
        if Jx is None:
            Jx = 32
        if Ky is None:
            Ky = 32

        # Create correlation Matrix in space, i.e. \phi_{j, k}(x ,y)
        space_corr_2d = np.array([[[[correlation(x, y, j, k, dx * (Nx - 1), dy * (Ny - 1)) for y in Y] for x in X] for k in range(Ky)] for j in range(Jx)])

        B = self.BM(s, t, dt, Jx, Ky)

        space_corr_reshaped = space_corr_2d.reshape(Jx*Ky, Nx, Ny)
        W = np.einsum('ij,jkl->ikl', B, space_corr_reshaped)

        return W

    def WN_space_time_2d_many(self, s, t, dt, a, b, dx, c, d, dy, num, Jx=None, Ky=None, correlation=None):

        return np.array(
            [self.WN_space_time_2d_single(s, t, dt, a, b, dx, c, d, dy, Jx=Jx, Ky=Ky, correlation=correlation) for _ in tqdm(range(num))])


    # Funciton for creating N random initial conditions of the form
    # a_{0,0} + \sum_{j=-px}^{j=px}\sum_{k=-py}^{k=py} a_{k,j}/(1+|k+j|^decay)*sin(k*\pi*x+j*\pi*y)
    def initial(self, N, X, Y, px=10, py=10, decay=2, scaling=1, Dirichlet=False):
        scale_x, scale_y = max(X) / scaling, max(Y) / scaling
        IC = np.zeros((N, len(X), len(Y)))
        SIN = np.array(
            [[[[np.sin(j * np.pi * x / scale_x / 2 - k * np.pi * y / scale_y / 2) / (
                        1 + np.abs(j) ** decay + np.abs(k) ** decay) for k in range(-py, py + 1)] for j in
               range(-px, px + 1)] for y in Y] for x in X])
        for i in range(N):
            sins = np.random.normal(size=(2 * px + 1) * (2 * py + 1))
            if Dirichlet:
                extra = 0
            else:
                extra = np.random.normal(size=1)
            for j in range(2 * px + 1):
                for k in range(2 * py + 1):
                    IC[i, :, :] = extra + sins[j + k * (2 * py + 1)] * SIN[:, :, j, k]
        return IC

    def WN_corr_2d(self, x, y, j, k, Lx, Ly):
        return np.sqrt(4 / (Lx * Ly)) * np.sin(j * np.pi * x / Lx) * np.sin(k * np.pi * y / Ly)

    def save_2d(self, W, name):
        np.save(name, W)

    def load_2d(self, name):
        return np.load(name)


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