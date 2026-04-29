# Portions of this code adapted from https://github.com/crispitagorico/torchspde
#  (originally from https://github.com/andrisger/Feature-Engineering-with-Regularity-Structures)
# Additional implementations and modifications by the authors of SPDEBench
#  to generate trajectories of Q-Wiener process with different truncation degree


import torch
import torch.nn.functional as F
import math
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

from timeit import default_timer

def get_twod_bj(dtref,J,a,alpha,device):
    """
    Alg 4.5 Page 443 in the book "An Introduction to Computational Stochastic PDEs"
    J: grid size. Eg. [64,64]
    """
    lambdax = 2 * np.pi * torch.cat([torch.arange(0,J[0]//2 +1,device=device), torch.arange(- J[0]//2 + 1,0,device=device)]) / a[0]
    lambday = 2 * np.pi * torch.cat([torch.arange(0,J[1]//2 +1,device=device), torch.arange(- J[1]//2 + 1,0,device=device)]) / a[1]
    lambdaxx, lambdayy = torch.meshgrid(lambdax,lambday)
    root_qj = torch.exp(- alpha * (lambdaxx ** 2 + lambdayy ** 2) / 2)
    bj = root_qj * np.sqrt(dtref) * J[0] * J[1] / np.sqrt(a[0] * a[1])
    return bj

def get_twod_dW(bj,kappa,M,device):
    """
    Alg 10.6 Page 444 in the book "An Introduction to Computational Stochastic PDEs"
    M: batch size
    """
    J = bj.shape
    if (kappa == 1):
        nn = torch.randn(M,J[0],J[1],2,device=device)
    else:
        nn = torch.sum(torch.randn(kappa,M,J[0],J[1],2,device=device),0)
    nn2 = torch.view_as_complex(nn)
    tmp = torch.fft.ifft2(bj*nn2,dim=[-2,-1])
    dW1 = torch.real(tmp)
    dW2 = torch.imag(tmp)
    return dW1,dW2


def process_tensor(x, dim, J=32, N=64):
    """
    Args:
        x: Input tensor of shape (batch, ?, J) or (batch, J, ?)
        dim: The dimension along which to process the vectors (1 or 2)
        J: Length of the original vectors, default 32
        N: Target length after padding, default 64
    Returns:
        Processed tensor with combined a and b, scaled by 4.
    """
    # Ensure dim is 1 or 2
    if dim not in [1, 2]:
        raise ValueError("dim must be 1 or 2")

    # Adjust dimensions so the vectors to process are along the last dimension
    if dim == 1:
        x = x.permute(0, 2, 1)  # Transpose to make vectors along the last dim

    # Process front part (first 17 elements)
    x_front = x[..., :17]  # Shape (B, J, 17)
    # Pad zeros after the first 17 elements to make length N
    v1 = F.pad(x_front, (0, N - 17))  # Pad on last dimension: (left=0, right=47)
    a = torch.fft.ifft(v1, dim=-1)

    # Process back part (last 15 elements)
    x_back = x[..., 17:32]  # Shape (B, J, 15)
    # Pad 17 zeros before and 32 zeros after
    v2 = F.pad(x_back, (17, 32))  # Pad on last dimension: (left=17, right=32)
    temp = torch.fft.ifft(v2, dim=-1)

    # Generate (-1)^k sequence
    k = torch.arange(N, device=x.device)
    sign = (-1) ** k  # Shape (N,)
    # Expand dimensions to match temp for broadcasting
    sign = sign.view(1, 1, N)
    # Multiply element-wise
    b = temp * sign

    # Combine a and b, then scale by 2
    result = (a + b)*2

    # Permute back if dim was 1
    if dim == 1:
        result = result.permute(0, 2, 1)  # Transpose back to original dimensions

    return result

def get_twod_dW_revised(bj, kappa, M, device):
    """
    M: batch size = 100
    J=[32,32]
    N=[64,64]
    """
    J = bj.shape
    if (kappa == 1):
        nn = torch.randn(M, J[0], J[1], 2, device=device)
    else:
        nn = torch.sum(torch.randn(kappa, M, J[0], J[1], 2, device=device), 0)
    nn2 = torch.view_as_complex(nn)
    B = bj * nn2

    # Method 1: (False)
    # tmp = batch_interpolated_2d_idft(B)

    # Method 2: (True, hopefully)
    tmp = process_tensor(B, dim=1, J=32, N=64)  # [batch,J,J] -> [batch,N,J]
    tmp = process_tensor(tmp, dim=2, J=32, N=64)  #           -> [batch,N,N]

    dW1 = torch.real(tmp)
    dW2 = torch.imag(tmp)
    return dW1, dW2


class GaussianRF(object):

    def __init__(self, dim, size, alpha=2, tau=3, sigma=None, boundary="periodic", device=None):

        self.dim = dim
        self.device = device

        if sigma is None:
            sigma = tau**(0.5*(2*alpha - self.dim))

        k_max = size//2

        if dim == 1:
            k = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                           torch.arange(start=-k_max, end=0, step=1, device=device)), 0)

            self.sqrt_eig = size*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0] = 0.0

        elif dim == 2:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,1)

            k_x = wavenumers.transpose(0,1)
            k_y = wavenumers

            self.sqrt_eig = (size**2)*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k_x**2 + k_y**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0,0] = 0.0

        elif dim == 3:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,size,1)

            k_x = wavenumers.transpose(1,2)
            k_y = wavenumers
            k_z = wavenumers.transpose(0,2)

            self.sqrt_eig = (size**3)*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k_x**2 + k_y**2 + k_z**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0,0,0] = 0.0

        self.size = []
        for j in range(self.dim):
            self.size.append(size)

        self.size = tuple(self.size)

    def sample(self, N):

        coeff = torch.randn(N, *self.size, 2, device=self.device)

        coeff[...,0] = self.sqrt_eig*coeff[...,0]
        coeff[...,1] = self.sqrt_eig*coeff[...,1]

        u = torch.fft.ifftn(torch.view_as_complex(coeff), dim=[1,2]).real

        return u