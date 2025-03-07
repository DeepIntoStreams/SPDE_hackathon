import numpy as np
import scipy.io
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from data_gen.src.Noise import Noise
from data_gen.src.general_solver import smooth_corr, general_1d_solver

n = 1200 # number of solutions
Nx, Nt = 2**7, 10**2  # number of observations in space and time
dx, dt = 1./Nx, 1./Nt  # space-time discretization
a, b, s, t = 0, 1, 0, 1 # space-time boundaries

u0 = np.array([[x*(1-x) for x in np.linspace(a,b,Nx+1)[:-1]] for _ in range(n)]) # initial condition

r = 4 # Creates r/2 spatially smooth noise
corr = lambda x, j, a : smooth_corr(x, j, a, r + 1.001)
W_smooth = Noise().WN_space_time_many(s, t, dt*0.1, a, b, dx, n, correlation = corr)

W_smooth = W_smooth[:,::10,:]

L_kdv = [0,0,1e-3,-0.1,0]
mu_kdv = lambda x: 0
sigma_kdv = lambda x: 1

KdV, _, _ = general_1d_solver(L_kdv, u0, W_smooth[:,:,:-1], mu = mu_kdv, sigma = sigma_kdv, Burgers= -6)

W = W_smooth.transpose(0,2,1)
soln = KdV.transpose(0,2,1)

O_X, O_T = np.linspace(0,1,W.shape[-2]), np.linspace(0,1,W.shape[-1])

save_dir = 'results/data_kdv/'
os.makedirs(save_dir, exist_ok=True)
scipy.io.savemat('results/data_kdv/kdv_xi_{}.mat'.format(n), mdict={'X':O_X, 'T':O_T, 'W': W, 'sol': soln})

