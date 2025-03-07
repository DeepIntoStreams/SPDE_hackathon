import scipy.io
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from data_gen.src.Noise import Noise
from data_gen.src.SPDEs import SPDE

n = 1200 # number of solutions
dx, dt = 1./128, 1./1000 # space-time increments
a, b, s, t = 0, 1, 0, 0.05 # space-time boundaries

ic = lambda x: x*(1-x) # initial condition (fixed)

# Another kind of initial condition (varying)
# X_ = np.linspace(-0.5,0.5,129)
# ic_ = Noise().initial(n, X_, scaling = 1) # one cycle
# ic = 0.1*(ic_-ic_[:,0,None]) + ic(O_X)

mu = lambda x: 3*x-x**3 # drift
sigma = lambda x: 1 # additive diffusive term

O_X, O_T = Noise().partition(a,b,dx), Noise().partition(s,t,dt) # space grid O_X and time grid O_T
W = Noise().WN_space_time_many(s, t, dt, a, b, dx, n) # create realizations of space time white noise
Soln_add = SPDE(BC = 'P', IC = ic, mu = mu, sigma = sigma).Parabolic(0.1*W, O_T, O_X) # solve parabolic equation

W = W.transpose(0,2,1)
soln = Soln_add.transpose(0,2,1)

save_dir = 'results/Phi41+/'
os.makedirs(save_dir, exist_ok=True)
scipy.io.savemat('results/Phi41+/Phi41+_xi_{}.mat'.format(n), mdict={'X':O_X, 'T':O_T, 'W': W, 'sol': soln})
