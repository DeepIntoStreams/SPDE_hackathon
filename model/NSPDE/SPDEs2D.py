import numpy as np
import pandas as pd
import math
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt
from scipy.fftpack import dst, idst


class SPDE2D():

    def __init__(self, Type='P', IC=lambda x, y: 0, IC_t=lambda x, y: 0, mu=None, sigma=1, BC='P', eps=1, T=None, X=None, Y=None):
        self.type = Type  # write down an elliptic ("E") or parabolic ("P") type
        self.IC = IC  # Initial condition for the parabolic equations
        self.IC_t = IC_t  # Initial condition for the time derivative
        self.mu = mu  # drift term in case of Parabolic or noise free term in case of Elliptic
        self.sigma = sigma  # sigma(u)*xi
        self.BC = BC  # Boundary condition 'D' - Dirichlet, 'N' - Neuman, 'P' - periodic
        self.eps = eps  # viscosity
        self.X = X  # discretization of space (O_X space)
        self.Y = Y  # discretization of space (O_Y space)
        self.T = T  # discretization of time (O_T space)

    def vectorized(self, f, vec):  # vectorises non-linearity and applies it to a certain vector
        if f is None:
            return 0
        if type(f) in {float, int}:
            return f
        return np.vectorize(f)(vec)

    def vectorized_2d(self, f, vec1, vec2):  # vectorises non-linearity and applies it to a certain vector
        v1, v2 = np.meshgrid(vec1, vec2)
        if f is None:
            return 0
        if type(f) in {float, int}:
            return f
        return np.vectorize(f)(v1, v2)
    
    def mu_(self, vec):
        if self.mu is None:
            return 0
        return self.vectorized(self.mu, vec)
    
    def sigma_(self, vec):
        if type(self.sigma) in {float, int}:
            return self.sigma
        return self.vectorized(self.sigma, vec)


    def initialization(self, W, T, X, Y, diff=True):

        dx, dy, dt = X[1] - X[0], Y[1] - Y[0], T[1] - T[0]
        # If only one noise given, reshape it to a 3d array with one noise.
        if len(W.shape) == 3:
            W.reshape((1, W.shape[0], W.shape[1], W.shape[2]))
        if diff:
            # Outups dW
            dW = np.zeros(W.shape)
            dW[:, 1:, :, :] = np.diff(W, axis=1)
        else:
            # Outputs W*dt
            dW = W * dt

        return dW, dx, dy, dt
    

    def partition(self, a, b, dx):  # makes a partition of [a,b] of equal sizes dx
        return np.linspace(a, b, int((b - a) / dx) + 1)

    def Solve(self, W):
        # if self.type == "E" or self.type == "Elliptic":
        #     return self.Elliptic(W)
        if self.type == "P" or self.type == "Parabolic":
            return self.Parabolic(W)
        # if self.type == "W" or self.type == "Wave":
        #     return self.Wave(W)
        # if self.type == "B" or self.type == "Burgers":
        #     return self.Burgers(W)
    

    # Calculat the matirx F = u_n + \mu(u_n)*dt + \sigma(\mu_n)*dW_n
    def Matrix_F(self, W, un, n, T=None, X=None, Y=None, diff=True):
        # n: the time step
        # un: u_n
        # Extract specae-time increments and dW
        dW, dx, dy, dt = self.initialization(W, T, X, Y, diff)

        F = np.zeros((W.shape[0], W.shape[2] , W.shape[3] ))
        for i in range(W.shape[0]):
            F[i, :, :] = un[i, :, :] + self.mu_(un[i, :, :]) * dt + self.sigma_(un[i, :, :]) * dW[i, n, :, :]

        return F

    
    # Definr the matrix A := \Delta_x * dt / dx ** 2
    def Matrix_A (self, W, T, X, Y):
        J = len(X)
        dW, dx, dy, dt = self.initialization(W, T, X, Y)
        A = np.diag(-2 * np.ones(J)) + np.diag(np.ones(J - 1), k=1) + np.diag(np.ones(J - 1), k=-1)
        A[0, J-1] = 1
        A[J-1, 0] = 1 
        A = A * dt / dx ** 2
        return A
    
    # Definr the matrix B := \Delta_y * dt / dy ** 2
    def Matrix_B (self, W, T, X, Y):
        K = len(Y)
        dW, dx, dy, dt = self.initialization(W, T, X, Y)
        B = np.diag(-2 * np.ones(K)) + np.diag(np.ones(K - 1), k=1) + np.diag(np.ones(K - 1), k=-1)
        B[0, K-1] = 1
        B[K-1, 0] = 1 
        B = B * dt / dy ** 2
        return B


    # Calculate u_{n+1} = F(u_n) + A * u_n + u_n * B, i.e. u_{n+1} = u_n + dt * \Delta_x(u_n) + dt * \Delta_y(u_n) + \mu(u_n)*dt + \sigma(\mu_n)*dW_n
    def Time_step(self, W, T, X, Y, dt, un, n): 
        
        A = self.Matrix_A(W, T, X, Y)
        B = self.Matrix_B(W, T, X, Y)
        F = self.Matrix_F(W, un, n, T, X ,Y)

        U = np.zeros((un.shape[0], un.shape[1], un.shape[2]))
        for i in range(un.shape[0]):
            U[i, :, :] = np.dot(A, un[i, :, :]) + np.dot(un[i, :, :], B)

        U += F
        return U
    

    # Calculate v_{n+1} = v_n + dt * \Delta_x(v_n) + dt * \Delta_y(v_n) + (v_n^3 - 3*v_n)*dt:
    def Renormalization_Time_step(self, W, T, X, Y, dt, un, n): 
        
        A = self.Matrix_A(W, T, X, Y)
        B = self.Matrix_B(W, T, X, Y)

        U = np.zeros((un.shape[0], un.shape[1], un.shape[2]))

        for i in range(un.shape[0]):
            U[i, :, :] = np.dot(A, un[i, :, :]) + np.dot(un[i, :, :], B)

        U += self.mu_(un) * dt + un[i, :, :]

        return U
    

    # Solves 2D Parabolic semilinear SPDEs, given numpy array of several noises, space time domain and initial conditions
    def Parabolic(self, W, T=None, X=None, Y=None, diff=True):

        # Extract specae-time increments and dW
        dW, dx, dy, dt = self.initialization(W, T, X, Y, diff)

        Solution = np.zeros(shape=W.shape)

        # define initial conditions
        if type(self.IC) is np.ndarray:
            if self.IC.shape == (W.shape[0], len(X), len(Y)):  # if initial conditions are given
                IC = self.IC
            else:
                IC = np.array([self.IC for _ in range(W.shape[0])])  # one initial condition
        else:
            initial = self.vectorized_2d(self.IC, X, Y)  # setting initial condition for one solution
            IC = np.array([initial for _ in range(W.shape[0])])  # for every solution

        # Initialize
        Solution[:, 0, :, :] = IC

        # T(u_{n+1}) = F, where T(U_{n+1}) = U_{n+1} - dt*\Delta*U_{n+1} - dt*U_{n+1}*\Delta, F = U_n + \mu(U_n)*dt + \sigma(U_n)*dW_n
        # u_{n+1} = T^{-1}(F)
        # Solve equations in paralel for every noise/IC simultaneosly
        for i in tqdm(range(1, len(T))):
            Solution[:, i, :, :] = self.Time_step(W, T, X, Y, dt, Solution[:, i - 1, :, :], i-1)

        return Solution.astype('float32')
    
    
    # Solve the equation \partial_t X_{\epsilon} - \Delta X_{\epsilon} = \xi_{\epsilon} with initial condition X_{\epsilon}(0,x,y) = v(0) = 0
    def FT_solver(self, W, T, X, Y, diff=True):
        # u0 - initial condition
        # W - space time noise
        # X - discretization of space (O_X space)
        # Y - discretization of space (O_Y space)
        # T - discretization of time (O_T space)
    
        # space time grid
        Nx, Ny =  W.shape[2], W.shape[3]
        dW, dx, dy, dt = self.initialization(W, T, X, Y, diff)
        J = len(X)
        K = len(Y)
        Lx = X[J-1] - X[0]
        Ly = Y[K-1] - Y[0]

        lap = - np.array([[2*Nx**2*np.cos(2*np.pi*k1/(Nx*Lx)) + 2*Ny**2*np.cos(2*np.pi*k2/(Ny*Ly)) for k2 in range(Ny)] for k1 in range(Nx)]) + 2*Ny**2 + 2*Nx**2 
        lap[0,0] = 1
        lap_ = 0.5*dt*self.eps*lap
    
        Solution = np.zeros(shape=W.shape)

        # Initialize: v(0) = 0
        Solution[:, 0, :, :] = np.zeros((W.shape[0], W.shape[2], W.shape[3]))
        IC = np.zeros((W.shape[0], W.shape[2], W.shape[3]))

        w = np.fft.fft2(IC, axes = [-2,-1])
    
        for i in tqdm(range(1, W.shape[1])):

            # non-linearities in the physical space
            RHS_noise = self.sigma_(Solution[:,i-1,:,:])*dW[:,i,:,:] if self.sigma_ is not None else dW[:,i,:,:]
        
            # updating the Fourier transform for the next time step
            w = ( (1-lap_) * w + np.fft.fft2(RHS_noise, axes = [-2,-1]) )/(1+lap_)
        
            # Going back to the physiscal space
            Solution[:,i,:,:] = np.fft.ifft2(w, axes = [-2,-1]).real
        
        return Solution
    

    # Computing the expectation of X_{\epsilon}^2 by Monte Carlo
    def MC(self, X_eps):

        num = X_eps.shape[0]
        T = X_eps.shape[1]
        a_eps = np.array([np.sum(X_eps[:, t, :, :] ** 2, axis=0) / num for t in range(T)])

        return a_eps
    
    # Solves 2D Singular SPDEs by renormalization, given numpy array of several noises, space time domain and initial conditions
    # dv = \Delta v * dt -:u^3: dt + 3*v dt with v(0) = 0, where :u^3: = v^3 + 3*v^2*X_{\epsilon} + 3*v*:X_{\epsilon}^2: + :X_{\epsilon}^3:
    # :X_{\epsilon}^2: = X_{\epsilon}^2 - a_{\epsilon}
    # :X_{\epsilon}^3: = X_{\epsilon}^3 - 3*X_{\epsilon}*a_{\epsilon}
    def Renormalization(self, W, T=None, X=None, Y=None, diff=True):

        # Extract specae-time increments and dW
        dW, dx, dy, dt = self.initialization(W, T, X, Y, diff)

        # X_{\epsilon}
        X_eps = self.FT_solver(W, T, X, Y)

        # a_{\epsilon}
        a_eps = self.MC(X_eps)

        # :X_{\epsilon}^2:
        Xsquare = np.zeros(shape=W.shape)
        for i in range(W.shape[0]):
            Xsquare[i, :, :, :] = X_eps[i, :, :, :] ** 2 - a_eps

        # :X_{\epsilon}^3:
        Xcubic= np.zeros((W.shape))
        for i in range(W.shape[0]):
            Xcubic[i, :, :, :] = X_eps[i, :, :, :] ** 3 - 3 * a_eps * X_eps[i, :, :, :]

        Solution = np.zeros(shape=W.shape)

        # define initial conditions
        if type(self.IC) is np.ndarray:
            if self.IC.shape == (W.shape[0], len(X), len(Y)):  # if initial conditions are given
                IC = self.IC
            else:
                IC = np.array([self.IC for _ in range(W.shape[0])])  # one initial condition
        else:
            initial = self.vectorized_2d(self.IC, X, Y)  # setting initial condition for one solution
            IC = np.array([initial for _ in range(W.shape[0])])  # for every solution

        # Initialize
        Solution[:, 0, :, :] = IC

        # T(u_{n+1}) = v^3 + 3*v + 3*v^2*X_{\epsilon} + 3*v*:X_{\epsilon}^2: + :X_{\epsilon}^3, where T(u_{n+1}) = u_{n+1} - dt*\Delta*U_{n+1} - dt*U_{n+1}*\Delta, F = U_n + \mu(U_n)*dt + \sigma(U_n)*dW_n
        # Solve equations in paralel for every noise/IC simultaneosly
        for i in tqdm(range(1, len(T))):
            Solution[:, i, :, :] = self.Renormalization_Time_step(W, T, X, Y, dt, Solution[:, i - 1, :, :], i-1) - (3 * Solution[:, i - 1, :, :] ** 2 * X_eps[:, i - 1, :, :] + 3 * Solution[:, i - 1, :, :] * Xsquare[:, i - 1, :, :] + Xcubic[:, i - 1, :, :]) * dt

        # u = v + X
        Solution += X_eps

        return Solution.astype('float32')