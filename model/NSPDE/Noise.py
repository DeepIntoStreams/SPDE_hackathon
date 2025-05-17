# Adapted from https://github.com/crispitagorico/torchspde
# Modified for current implementation by the authors of SPDEBench
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
import torch

def partition(a, b, dx):  # makes a partition of [a,b] of equal sizes dx
    return np.linspace(a, b, int((b - a) / dx) + 1)


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
    # X here is points in space as in SPDE function.
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
        Nt = len(T)

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

    def WN_space_time_2d_many(self, s, t, dt, a, b, dx, c, d, dy, num, Jx=None, Ky=None,  correlation=None):

        return np.array(
            [self.WN_space_time_2d_single(s, t, dt, a, b, dx, c, d, dy, Jx, Ky, correlation=correlation) for _ in tqdm(range(num))])

    def WN_corr_2d(self, x, y, j, k, Lx, Ly):
        return np.sqrt(4 / (Lx * Ly)) * np.sin(j * np.pi * x / Lx) * np.sin(k * np.pi * y / Ly)

    def save_2d(self, W, name):
        np.save(name, W)

    def load_2d(self, name):
        return np.load(name)