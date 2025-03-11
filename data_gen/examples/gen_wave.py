import os
import os.path as osp
import sys

current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(osp.join(current_directory, "..",".."))

import numpy as np
import scipy.io
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from data_gen.src.Noise import Noise
from data_gen.src.SPDEs import SPDE

n = 1000 # Number of realizations
dx, dt = 0.01, 0.001 #space-time increments
a, b, s, t = 0, 1, 0, 1 # space-time boundaries

O_X, O_T = Noise().partition(a, b, dx), Noise().partition(s, t, dt) # space grid O_X and time grid O_T

W = Noise().WN_space_time_many(s, t, dt, a, b, dx, n) # Create realizations of space time white noise

ic = lambda x : np.sin(2*np.pi*x) # initial condition
ic_t = lambda x : x*(1-x) # initial speed
mu = lambda x: np.cos(np.pi*x)+x**2 # drift
sigma = lambda x : x # diffusion

# solve wave equation
Wave_soln = SPDE(Type = 'W', BC = 'P', T = O_T, X = O_X, IC = ic, IC_t = ic_t, mu = mu, sigma = sigma).Wave(W)

W = W.transpose(0,2,1)
soln = Wave_soln.transpose(0,2,1)

save_dir = 'data_gen/results/data_wave/'
os.makedirs(save_dir, exist_ok=True)
scipy.io.savemat('data_gen/results/data_wave/wave_xi_{}.mat'.format(n), mdict={'X':O_X, 'T':O_T, 'W': W, 'sol': soln})

