# Adapted from https://github.com/crispitagorico/torchspde
# Modified for current implementation by the authors of SPDEBench

import numpy as np
from tqdm import tqdm


def general_1d_solver(L, u0, W, mu, sigma=lambda x: 1, T=1, X=1, Burgers=0, KPZ=0, compl=False, D=None):
    # L = [c_0, ... c_3, c_4] corresponds to a differential operator L = \sum c_i \partial^i_x
    # in the parabolic operator (\partial_t - L)

    # NB: including c_0 != 0 allows to ignore it in the non-linearity: taking mu(u) = -u^3 + 3u and c_0 = 0
    # is equivalent to taking mu(u) = -u^3 and c_0 = 3

    # u0 - initial condition
    # xi - space time noise
    # mu - drift term of the equation
    # sigma - diffusion term of the equation
    # T - time horizon
    # X - space domain is [0,X]
    # Burgers - coefficient in front of the Burger's term u*\partial_x u in the equation
    # KPZ - coefficient in front of the KPZ term (\partial_x u)**2 in the equation
    # D - Derivative operator applied to the drift on the RHS

    # E.g. in Cahn-Hilliard L = [0,0,0,0,-1], D = [0,0,1,0,0] since the equation is

    # (\partial_t + \Delta^2) u = \Delta (- u^3) + \xi

    # Check operator conditoins. Must have c_4 < 0 or c_2 > 0 if c_4 = 0
    if (type(L[-1]) != complex) and (L[-1] > 0) or (L[-1] == 0 and type(L[2]) != complex and L[2] < 0):
        print("Differential Operator is not Elliptic.")
        return

    # space time grid
    M, N = W.shape[-1], W.shape[-2] - 1,
    dt = T / N

    # Time deirvative of the noise

    if len(W.shape) > 2:
        dW = np.zeros(W.shape)
        dW[:, 1:, :] = np.diff(W, axis=1)
    else:
        dW = np.zeros((1, W.shape[0], W.shape[1]))
        dW[:, 1:, :] = np.diff(W, axis=0)

    K = np.arange(M)  # Fourier space
    # Space derivatives in Fourier space. These are analogues of central finite difference in Fourier Space
    # All these give an error of O(N^{-2}) if n-th derivative is approximated by (2*np.pi*K)^n

    dx = M / X * 1j * np.sin(2 * np.pi * K / M)
    d2x = 2 * (M / X) ** 2 * (np.cos(2 * np.pi * K / M) - 1)
    d3x = dx * d2x
    d4x = 2 * (M / X) ** 4 * (np.cos(4 * np.pi * K / M) - 4 * np.cos(2 * np.pi * K / M) + 3)

    Lk = L[0] + L[1] * dx + L[2] * d2x + L[3] * d3x + L[4] * d4x  # Full differential operator in the Fourier space
    if D is not None:
        Dk = D[0] + D[1] * dx + D[2] * d2x + D[3] * d3x + D[4] * d4x
    else:
        Dk = 1

    if compl:
        soln = np.zeros(dW.shape) + 1j * np.zeros(dW.shape)
    else:
        soln = np.zeros(dW.shape)

    soln[:, 0, :] = u0  # set the initial condition

    w = np.fft.fft(u0, axis=-1)

    for i in tqdm(range(1, N + 1)):

        Extra_nonlinearity = 0
        if Burgers != 0 or KPZ != 0:  # if Burgers or KPZ is present compute space derivative. M/X = (dx)^{-1}
            diff = np.zeros((soln.shape[0], M))
            diff[:, 1:] = np.diff(soln[:, i - 1, :], axis=1)
            diff[:, 0] = (soln[:, i - 1, 0] - soln[:, i - 1, -1]) * M / X
        if Burgers != 0:  # add Burgers nonlnearity u*(\partial_x u)
            Extra_nonlinearity += Burgers * soln[:, i - 1, :] * diff
        if KPZ != 0:  # add KPZ nonlinearity (\partial_x u)^2
            Extra_nonlinearity += KPZ * diff * diff

        # non-linearities in the physical space
        RHS_drift = np.vectorize(mu)(soln[:, i - 1, :]) * dt
        RHS_extra = Extra_nonlinearity * dt
        RHS_noise = np.vectorize(sigma)(soln[:, i - 1, :]) * dW[:, i, :]

        # updating the Fourier transform for the next time step
        w = ((1 + 0.5 * Lk * dt) * w + np.fft.fft(RHS_noise + RHS_extra, axis=-1) + Dk * np.fft.fft(RHS_drift,
                                                                                                    axis=-1)) / (
                        1 - 0.5 * Lk * dt)

        # Going back to the physiscal space
        if compl:
            soln[:, i, :] = np.fft.ifft(w, axis=-1)
        else:
            soln[:, i, :] = np.fft.ifft(w, axis=-1).real

    return soln, np.linspace(0, T, N + 1), np.linspace(0, X, M + 1)[:-1]