# Partially adapted from https://github.com/RazliTamir/HaarApproximation


import math
import numpy as np
import pandas as pd
from typing import Iterator, TypeAlias
from random import choice
import matplotlib.pyplot as plt
from numpy.typing import NDArray
import sympy as sp

x = sp.Symbol('x')  # x axis
j = sp.Symbol('j')  # slice index
d = sp.Symbol('d')  # depth index
f = sp.Function('f')(x)  # f(x)  # pylint:disable = not-callable


__LOWER_BOUND: sp.Expr = j / (2 ** (d - 1))  # slice lower bound
__MIDDLE_POINT: sp.Expr = (j + (1 / 2)) / (2 ** (d - 1))  # slice middle
__UPPER_BOUND: sp.Expr = (j + 1) / (2 ** (d - 1))  # slice upper bound

# used for calculating amplitude
AMP_EXP: sp.Expr = (
    (2 ** (d - 1)) * (
        sp.integrate(f, (x, __LOWER_BOUND, __MIDDLE_POINT)) -
        sp.integrate(f, (x, __MIDDLE_POINT, __UPPER_BOUND))
    )
)
AMP_EXP_AT_0: sp.Expr = sp.integrate(f, (x, 0, 1))


def linspace_from_resolution(res: int):
    """initialize linspace with resolution (number of elements) of 2^res"""
    return np.linspace(0, 1, 2 ** res)


def plot(haar_: 'Haar'):
    """shorthand plot function"""
    lin_space = linspace_from_resolution(haar_.max_depth)
    plt.step(lin_space, haar_.array, where='mid')
    plt.show()


class Haar:
    """1 dimensional haar approximation"""

    # shorthand types
    TData: TypeAlias = NDArray[np.float64]
    PosNegTuple: TypeAlias = tuple[TData, TData]

    def __init__(self, max_depth: int):
        # initialize array with resolution (number of elements) of 2^res
        self.max_depth = max_depth
        self.array = np.zeros(2 ** max_depth)
        self.slices = self.__init_slice_cache(max_depth)

    @classmethod
    def from_amplitudes(cls, amps: dict[tuple[int, int], np.float64], max_depth: int):
        """initialize using known slices->amplitude map"""
        out = cls(max_depth)
        for (depth, index), amp in amps.items():
            amp = np.float64(amp)
            if depth == 0:
                # amplitude is uniform bias at depth = 0
                out.array += amp
                continue
            pos, neg = out.slices[depth][index]
            pos += amp
            neg -= amp
        return out

    @classmethod
    def from_expression(cls, exp: sp.Expr, max_depth: int):
        """initialize using max depth (for resolution), and expression to approximate"""
        amps: dict[tuple[int, int], np.float_] = {
            (depth, index): get_amplitude(exp, depth, index)
            for depth, index in iter_depth(max_depth)
        }
        return cls.from_amplitudes(amps, max_depth)

    @classmethod
    def random(cls, max_depth: int):
        """generates random haar array from amplitudes taken from {-1, 1}"""
        amps = {
            (depth, index): np.float64(choice([-1, 1]))
            for depth, index in iter_depth(max_depth)
        }
        return cls.from_amplitudes(amps, max_depth)

    def __init_slice_cache(self, max_depth: int):
        """
        depth -> slice index -> (pos, neg)
        mapping initializer
        """
        slices: dict[int, dict[int, Haar.PosNegTuple]] = {}
        for depth in range(1, max_depth + 1):
            slices[depth] = {}
            if depth == 1:
                pos, neg, *_ = np.split(self.array, 2)
                slices[depth][0] = (pos, neg)
                continue
            parts = 2 ** (depth - 1)
            partitions = np.split(self.array, parts)
            for index, partition in enumerate(partitions):
                pos, neg, *_ = np.split(partition, 2)
                slices[depth][index] = (pos, neg)
        return slices


def get_amplitude(func: sp.Expr, depth: int, index: int):
    """get amplitude for slice at given depth and index, for a given expression"""
    if depth < 0:
        raise ValueError(f"depth can't be below 0: {depth=}")
    if not 0 <= index <= (2 ** depth):
        raise ValueError(
            f"incorrect range: 0 <= ({index=}) <= ({2 ** depth=})"
        )
    exp = (
        AMP_EXP.subs({d: depth, j: index, f: func, })
        if depth else AMP_EXP_AT_0.subs({f: func})
    )
    return exp.evalf()


def iter_depth(max_depth: int) -> Iterator[tuple[int, int]]:
    """iterates through range of depth and indeces, for a given maximal depth"""
    if max_depth < 0:
        raise ValueError(f"depth cant be below 0: {max_depth=}")
    yield (0, 0)
    if max_depth == 0:
        return
    yield (1, 0)
    if max_depth == 1:
        return
    for depth in range(2, max_depth + 1):
        for index in range(0, 2 ** (depth - 1)):
            yield (depth, index)


class Noise():

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

    # Create space-time noise. White in time and with some correlation in space.
    # See Example 10.31 in "AN INTRODUCTION TO COMPUTATIONAL STOCHASTIC PDES" by Lord, Powell, Shardlow
    # X here is points in space as in SPDE1 function.
    def WN_space_time_single(self, s, t, dt, a, b, dx, J=None, correlation=None, numpy=True):

        # If correlation function is not given, use space-time white noise correlation fucntion
        if correlation is None:
            correlation = self.WN_corr

        T, X = self.partition(s, t, dt), self.partition(a, b, dx)  # time points, space points,
        N = len(X)
        if J == None:
            J = 2

        # Create correlation Matrix in space
        # space_corr = np.array([[correlation(x, j, dx * (N - 1)) for j in range(J+1)] for x in X])
        B = self.BM(s, t, dt, J)
        num_time_steps = B.shape[0]  # N_{t} = 51
        J_depth = int(math.log2(J))  # it should be: 2**max_depth = J
        Nx_depth = int(math.log2(N))
        max_depth = max(J_depth, Nx_depth)
        W = []
        for time_step in range(num_time_steps):
            amps = {}
            for depth, index in iter_depth(max_depth):
                if depth > J_depth:
                    val = 0.0
                elif depth == 0:
                    val = B[time_step, 0]
                else:
                    val = (2 ** ((depth - 1) / 2)) * B[time_step, 2 ** (depth - 1) + index]
                amps[(depth, index)] = val
            haar_1d = Haar.from_amplitudes(amps, max_depth)
            W.append(haar_1d.array)  # its shape is (2**max_depth, )
        W = np.pad(
            np.array(W),
            pad_width=((0, 0), (1, 0)),
            mode='constant',
            constant_values=0
        )  # its shape is (num_time_steps, 2**max_depth+1). We want it to be (num_time_steps, 2**Nx_depth+1)
        if W.shape[1]>N:
            W = W[:,::(2**J_depth//2**Nx_depth)]
        return W

        # if numpy:
        #     return np.dot(B, space_corr.T)

        # return pd.DataFrame(np.dot(B, space_corr), index=T, columns=X)

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

