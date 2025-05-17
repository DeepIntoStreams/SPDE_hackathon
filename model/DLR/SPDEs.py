# Adapted from https://github.com/sdogsq/DLR-Net

import numpy as np
import pandas as pd
import math
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt

class SPDE():
    
    def __init__(self, Type = 'P', IC = lambda x: 0, IC_t = lambda x: 0, mu = None, sigma = 1, BC = 'P', eps = 1, T = None, X = None):
        self.type = Type # write down an elliptic ("E") or parabolic ("P") type
        self.IC = IC # Initial condition for the parabolic equations
        self.IC_t = IC_t # Initial condition for the time derivative
        self.mu = mu # drift term in case of Parabolic or noise free term in case of Elliptic
        self.sigma = sigma # sigma(u)*xi
        self.BC = BC #Boundary condition 'D' - Dirichlet, 'N' - Neuman, 'P' - periodic
        self.eps = eps # viscosity
        self.X = X # discretization of space (O_X space)
        self.T = T # discretization of time (O_T space)
        
    def vectorized(self, f, vec): # vectorises non-linearity and applies it to a certain vector
        if f is None:
            return 0
        if type(f) in {float, int}:
            return f
        return np.vectorize(f)(vec)
    
    def mu_(self, vec):
        if self.mu is None:
            return 0
        return self.vectorized(self.mu, vec)
    def sigma_(self, vec):
        if type(self.sigma) in {float, int}:
            return self.sigma
        return self.vectorized(self.sigma, vec)
    
    def nghbr_in_space(self, vec, pos, N): 
        #For a vector of length m x N subdivides it into m vectors and shifts each of them by pos. 
        return np.roll(vec.reshape(len(vec)//N, N), pos, axis = 1).flatten()
    
    def discrete_diff(self, vec, N, f = None, flatten = True, higher = True):
        a = vec.copy()
        if len(a.shape) == 1:
            a = a.reshape(len(vec)//N, N)
        if f is None:
            if higher: # central approximation of a dervative
                a[:,:-1] = (np.roll(a[:,:-1], -1, axis = 1) - np.roll(a[:,:-1], 1, axis = 1))/2
            else:
                a[:,:-1] = a[:,:-1] - np.roll(a[:,:-1], 1, axis = 1)
        else:
            # if a finction f given output d f(vec) / dx instead of d(vec)/dx
            if higher:
                a[:,:-1] = (self.vectorized(f, np.roll(a[:,:-1], -1, axis = 1)) - self.vectorized(f, np.roll(a[:,:-1], 1, axis = 1)))/2
            else:
                a[:,:-1] = self.vectorized(f, a[:,:-1]) - self.vectorized(f, np.roll(a[:,:-1], 1, axis = 1))
        a[:,-1] = a[:,0] # enforce periodic boundary condions
        if flatten:
            return a.flatten()
        return a
    
    def initialization(self, W, T, X, diff = True):
        
        dx, dt = X[1]-X[0], T[1]-T[0]
        # If only one noise given, reshape it to a 3d array with one noise.
        if len(W.shape) == 2:
            W.reshape((1,W.shape[0], W.shape[1]))
        if diff:
            # Outups dW
            dW = np.zeros(W.shape)
            dW[:,1:,:] = np.diff(W, axis = 1)
        else:
            # Outputs W*dt
            dW = W*dt
        
        return dW, dx, dt
        
        
    def Parabolic_Matrix(self, N, dt, dx, inverse = True): #(N+1)x(N+1) Matrix approximating (Id - eps * \Delta*dt)^{-1}
        # 'D' corresponds to Dirichlet, 'N' to Neuman, 'P' to periodic BC
        # Approximate sceletot of the Laplacian
        A = np.diag(-2 * np.ones(N + 1)) + np.diag(np.ones(N), k=1) + np.diag(np.ones(N), k=-1) 
        if self.BC == 'D': # if Dirichlet BC adjust # u(X[0]) = u(X[N]) = 0
            A[0,0], A[0,1], A[1,0], A[-1,-1], A[-1,-2], A[-2,-1] = 0, 0, 0, 0, 0, 0
        if self.BC == 'N': # if Neuman BC adjust
            A[0,1], A[-1,-2] = 2, 2
        if self.BC == 'P':
            A[-1, 1], A[0, -2] = 1, 1
        
        if inverse:
            return  np.linalg.inv(np.identity(N + 1) - self.eps*dt * A / (dx ** 2))
        
        # Matrix approximation of eps * \Delta*dt
        return self.eps*dt * A / (dx ** 2)
    
    def KdV_Matrix(self, N, dt, dx, inverse = True): 
        # Matrix approximating or (Id-eps*\partial_{xxx}*dt)^{-1} or eps*\partial_{xxx}*dt for potential application 
        # of the KdV equation
        
        # Approximate sceletot of the Laplacian
        A = np.diag(-np.ones(N), k=1) + np.diag(np.ones(N), k=-1) + np.diag(np.ones(N-1)/2, k=2) + np.diag(-np.ones(N-1)/2, k=-2)

        A[0, -3], A[0, -2] = -0.5, 1
        A[1, -2] = -0.5
        A[-2, 1] = 0.5
        A[-1, 1], A[-1, 2] = -1, 0.5
        
        if inverse:
            return  np.linalg.inv(np.identity(N + 1) + self.eps*dt * A / (dx ** 3))
        return self.eps*dt * A / (dx ** 3)
    
    def partition(self, a,b, dx): #makes a partition of [a,b] of equal sizes dx
        return np.linspace(a, b, int((b - a) / dx) + 1)
    
    def Solve(self, W):
        if self.type == "E" or self.type == "Elliptic":
            return self.Elliptic(W)
        if self.type == "P" or self.type == "Parabolic":
            return self.Parabolic(W)
        if self.type == "W" or self.type == "Wave":
            return self.Wave(W)
        if self.type == "B" or self.type == "Burgers":
            return self.Burgers(W)
    
    # Solves 1D Parabolic semilinear SPDEs, given numpy array of several noises, space time domain and initial conditions
    def Parabolic(self, W, T = None, X = None, diff = True):
        if T is None: T = self.T
        if X is None: X = self.X
        
        # Extract specae-time increments and dW
        dW, dx, dt = self.initialization(W, T, X, diff)
        
        M = self.Parabolic_Matrix(len(X)-1, dt, dx).T #M = (I-\Delta*dt)^{-1}

        Solution = np.zeros(shape = W.shape)
        
        # define initial conditions
        if type(self.IC) is np.ndarray:
            if self.IC.shape == (W.shape[0], len(X)): #if initial conditions are given
                IC = self.IC
            else:
                IC = np.array([self.IC for _ in range(W.shape[0])]) # one initial condition
        else:
            initial = self.vectorized(self.IC, X) # setting initial condition for one solution
            IC = np.array([initial for _ in range(W.shape[0])]) #for every solution
        
        # Initialize
        Solution[:,0,:] = IC
        
        # Finite difference method.
        # u_{n+1} = u_n + (dx)^{-2} A*u_{n+1}*dt + mu(u_n)*dt + sigma(u_n)*dW_{n+1}
        # Hence u_{n+1} = (I - dt/(dx)^2 A)^{-1} (u_n + mu(u_n)*dt + sigma(u_n)*dW_{n+1})
        # Solve equations in paralel for every noise/IC simultaneosly
        for i in tqdm(range(1,len(T))):
            current = Solution[:,i-1,:] + self.mu_(Solution[:,i-1,:]) * dt + self.sigma_(Solution[:,i-1,:]) * dW[:,i,:]
            Solution[:,i,:] = np.dot(current.reshape((W.shape[0], len(X))), M)
            
        # Because Solution.iloc[i-1] and thus current is a vector of length len(noises)*len(X)
        # need to reshape it to matrix of the shape (W.shape[0], len(X)) and multiply on the right by the M^T (transpose).
        # M*(current.reshape(...)) does not give the correct value.
        
        
        return Solution
    
    # Solves 1D stochasrtic Wave equation, given numpy array of several noises and space time domain
    def Wave(self, W, T = None, X = None, diff = True):
        if T is None: T = self.T
        if X is None: X = self.X
        
        # Extract specae-time increments and dW
        dW, dx, dt = self.initialization(W, T, X, diff)
        
        A = self.Parabolic_Matrix(len(X)-1, dt, dx, inverse = False).T # eps*\Delta*dt
        
        # Arrays of solution and its time derivative
        Solution, Solution_t = np.zeros(shape = W.shape), np.zeros(shape = W.shape)
        
        # Define fixed initial conditions for every realizarion of the noise 
        initial, initial_t = self.vectorized(self.IC, X), self.vectorized(self.IC_t, X)

        # Initialize
        Solution[:,0,:] = np.array([initial for _ in range(W.shape[0])])
        Solution_t[:,0,:] = np.array([initial_t for _ in range(W.shape[0])])
        
        # Finite difference method.
        # Solve wave equation as a system for (Solution, Solition_t)
        for i in tqdm(range(1, len(T))):
        
            Solution_t[:,i,:] = Solution_t[:,i-1,:] + np.dot(Solution[:,i-1,:].reshape((W.shape[0], len(X))), A)  + self.mu_(Solution[:,i-1,:]) * dt + self.sigma_(Solution[:,i-1,:]) * dW[:,i,:]
            Solution[:,i,:] = Solution[:,i-1,:] + Solution_t[:,i,:]*dt
            
        return Solution
    
    # Solves 1D Burger's equation, given numpy array of several noises, space time domain and initial conditions
    def Burgers(self, W, lambd = 1, diff = True, T = None, X = None, KdV = False):    
        # Extract increments, columns, time points and noise names
        if T is None: T = self.T
        if X is None: X = self.X
        
        # Extract specae-time increments and dW
        dW, dx, dt = self.initialization(W, T, X, diff)
        
        # Burger's equation is solved only with periodic boundary conditions due to a slightly easier computation of
        # of spatial derivative. 
        
        self.BC = 'P' 
        if KdV: # If KdV equation is considered
            M = self.KdV_Matrix(len(X)-1, dt, dx).T
        else:
            M = self.Parabolic_Matrix(len(X)-1, dt, dx).T #M = (I-\Delta*dt)^{-1}

        Solution = np.zeros(shape = W.shape)
        
        # define initial conditions
    
        if type(self.IC) is np.ndarray:
            if self.IC.shape == (W.shape[0], len(X)): #if initial conditions are given
                IC = self.IC
            else:
                IC = np.array([self.IC for _ in range(W.shape[0])]) # one initial condition
        else:
            initial = self.vectorized(self.IC, X) # setting initial condition for one solution
            IC = np.array([initial for _ in range(W.shape[0])]) #for every solution
        
        # Initialize
        Solution[:,0,:] = IC
        
        # Finite difference method.
        # u_{n+1} = u_n + (dx)^{-2} A*u_{n+1}*dt - u_n(u^+_n - u_n)*dt/dx + sigma(u_n)*dW_{n+1}
        # Hence u_{n+1} = (I - dt/(dx)^2 A)^{-1} (u_n -u_n(u^+_n - u_n)*dt/dx + sigma(u_n)*dW_{n+1})
        for i in tqdm(range(1, len(T))):
            
            current = Solution[:,i-1,:] - lambd*Solution[:,i-1,:]*self.discrete_diff(Solution[:,i-1,:], len(X), flatten = False, higher = False)*dt/dx + self.sigma_(Solution[:,i-1,:]) * dW[:,i,:]
            Solution[:,i,:] = np.dot(current.reshape((W.shape[0], len(X))), M)
            
        return Solution
    
    # Function that integrates trees required for the model. It applies operator I in paralel to many functions
    # where I[f] is a solution to (\partial_t - eps*\Delta) I[f] = f with zero IC.
    
    def Integrate_Parabolic_trees(self, tau, done = {}, exceptions = {}, derivative = False):
    
        #extract the trees from dictionary which are not purely polyniomials and were not already integrated
        trees = [tree for tree in tau.keys() if 'I[{}]'.format(tree) not in done and 'I[{}]'.format(tree) not in exceptions] 

        dt, dx = self.T[1]-self.T[0], self.X[1]-self.X[0] 
        
        # approximate inverse of (I - Laplacian)
        M = self.Parabolic_Matrix(len(self.X)-1, dt, dx).T

        taus = np.array([tau[t] for t in trees])

        integrated = np.zeros(shape = (len(trees), len(self.T), len(self.X)))
        
        # Finite difference method.
        # Compute M*(integrated[i-1]+taus[i]). M^T (transpose) and reshaping for the same reason as in Parabolic_many
        for i in range(1,len(self.T)): 
            integrated[:,i,:] = np.dot((integrated[:,i-1,:] + taus[:,i,:] * dt).reshape((len(trees), len(self.X))), M)
        
        Jtau = {}
        for i, t in enumerate(trees): #update dictionary and return integrated taus.
            Jtau['I[{}]'.format(t)] = integrated[i]
            if derivative and "I'[{}]".format(t) not in exceptions:
                # If derivative is true include also functions of the form \partial_x I[f] that are denoted as I'[f]
                if derivative == 1 or derivative is True:
                    Jtau["I'[{}]".format(t)] = self.discrete_diff(integrated[i], len(self.X), flatten = False, higher = False)/dx
                else:
                    # centralised differentiation
                    Jtau["I'[{}]".format(t)] = self.discrete_diff(integrated[i], len(self.X), flatten = False, higher = True)/dx
        return Jtau
    
    # Function that integrates trees required for the model. It applies operator I in paralel to many functions
    # where I[f] is a solution to (\partial^2_t - eps*\Delta) I[f] = f with zero IC.
    
    def Integrate_Wave_trees(self, tau, done = {}, exceptions = {}, derivative = False):
        
        #extract the trees from dictionary which are not purely polyniomials and were not already integrated
        trees = [tree for tree in tau.keys() if 'I[{}]'.format(tree) not in done and 'I[{}]'.format(tree) not in exceptions] 

        dt, dx = self.T[1]-self.T[0], self.X[1]-self.X[0] 
        
        # approximate Laplacian
        A = self.Parabolic_Matrix(len(self.X)-1, dt, dx, inverse=False).T

        taus = np.array([tau[t] for t in trees])
        
        # Initialize placeholders for I[f] and \partial_t I[f]
        integrated = np.zeros(shape = (len(trees), len(self.T), len(self.X)))
        integrated_t = np.zeros(shape = (len(trees), len(self.T), len(self.X)))
        
        # Finite difference method.
        # Solve wave equation as a system for (integrated, integrated_t)
        for i in range(1,len(self.T)):
            
            integrated_t[:,i,:] = integrated_t[:,i-1,:] + np.dot(integrated[:,i-1,:].reshape((len(trees), len(self.X))), A) + taus[:,i,:] * dt
            integrated[:,i,:] = integrated[:,i-1,:] + integrated_t[:,i,:]*dt
            
        Jtau = {}
        for i, t in enumerate(trees): #update dictionary and return integrated taus.
            Jtau['I[{}]'.format(t)] = integrated[i]
            if derivative and "I'[{}]".format(t) not in exceptions:
                # If derivative is true include also functions of the form \partial_x I[f] that are denoted as I'[f]
                if derivative == 1 or derivative is True:
                    Jtau["I'[{}]".format(t)] = self.discrete_diff(integrated[i], len(self.X), flatten = False, higher = False)/dx
                else:
                    Jtau["I'[{}]".format(t)] = self.discrete_diff(integrated[i], len(self.X), flatten = False, higher = True)/dx
        return Jtau
    
    #To finish
    def Integrate_Parabolic_trees_2d(self, tau, done = {}, exceptions = {}, derivative = False):
    
        #extract the trees from dictionary which are not purely polyniomials and were not already integrated
        trees = [tree for tree in tau.keys() if 'I[{}]'.format(tree) not in done and 'I[{}]'.format(tree) not in exceptions] 

        dt, dx = self.T[1]-self.T[0], self.X[1,0]-self.X[0,0]
        taus = np.array([tau[t] for t in trees])

        integrated = np.zeros(shape = (len(trees), len(self.T), self.X.shape[0], self.X.shape[1]))
        
        # Finite difference method.
        # Compute M*(integrated[i-1]+taus[i]). M^T (transpose) and reshaping for the same reason as in Parabolic_many
        for i in range(1,len(self.T)): 
            integrated[:,i,:,:] = (integrated[:,i-1,:,:] + self.eps * self.Laplace_2d(integrated[:,i-1,:,:], dx, dt) + taus[:,i-1,:,:] * dt).reshape((len(trees), self.X.shape[0], self.X.shape[1]))
        
        Jtau = {}
        for i, t in enumerate(trees): #update dictionary and return integrated taus.
            Jtau['I[{}]'.format(t)] = integrated[i]
            if derivative and "I1[{}]".format(t) not in exceptions and "I2[{}]".format(t) not in exceptions and t[1] == "[":
                # If derivative is true include also functions of the form \partial_x I[f] that are denoted as I'[f]
                if derivative == 1 or derivative is True:
                    Jtau["I1[{}]".format(t)] = self.discrete_diff_2d(integrated[i], self.X.shape[0], axis = 1, flatten = False, higher = False)/dx
                    Jtau["I2[{}]".format(t)] = self.discrete_diff_2d(integrated[i], self.X.shape[0], axis = 2, flatten = False, higher = False)/dx
                else:
                    # centralised differentiation
                    Jtau["I1[{}]".format(t)] = self.discrete_diff_2d(integrated[i], self.X.shape[0], axis = 1, flatten = False, higher = True)/dx
                    Jtau["I2[{}]".format(t)] = self.discrete_diff_2d(integrated[i], self.X.shape[0], axis = 2, flatten = False, higher = True)/dx
        return Jtau

    def Laplace_2d(self, arr, dx, dt):
        if not isinstance(arr, np.ndarray):
            return 0
        out = np.zeros(arr.shape)
        out[:,1:-1,:] += np.diff(arr, 2, axis=1)
        out[:,:,1:-1] += np.diff(arr, 2, axis=2)
                
        if self.BC =='P':
            out[:,0,1:-1] = arr[:,1,1:-1] + arr[:,-1,1:-1] + arr[:,0,0:-2] + arr[:,0,2:] - 4*arr[:,0,1:-1]
            out[:,-1,1:-1] = arr[:,0,1:-1] + arr[:,-2,1:-1] + arr[:,-1,0:-2] + arr[:,-1,2:] - 4*arr[:,-1,1:-1]
            out[:,1:-1,0] = arr[:,1:-1,1] + arr[:,1:-1,-1] + arr[:,0:-2,0] + arr[:,2:,0] - 4*arr[:,1:-1,0]
            out[:,1:-1,-1] = arr[:,1:-1,0] + arr[:,1:-1,-2] + arr[:,0:-2,-1] + arr[:,2:,-1] - 4*arr[:,1:-1,-1]
            
        return out*dt/dx**2


    def discrete_diff_2d(self, vec, N, axis, f = None, flatten = True, higher = True):
        a = vec.copy()
        if len(a.shape) == 1:
            a = a.reshape(len(vec)//N, N)
        if axis == 1:
            if f is None:
                if higher: # central approximation of a dervative
                    a[:,:-1,:] = (np.roll(a[:,:-1,:], -1, axis = 1) - np.roll(a[:,:-1,:], 1, axis = 1))/2
                else:
                    a[:,:-1,:] = a[:,:-1,:] - np.roll(a[:,:-1,:], 1, axis = 1)
            else:
                # if a finction f given output d f(vec) / dx instead of d(vec)/dx
                if higher:
                    a[:,:-1,:] = (self.vectorized(f, np.roll(a[:,:-1,:], -1, axis = 1)) - self.vectorized(f, np.roll(a[:,:-1,:], 1, axis = 1)))/2
                else:
                    a[:,:-1,:] = self.vectorized(f, a[:,:-1,:]) - self.vectorized(f, np.roll(a[:,:-1,:], 1, axis = 1))
            a[:,-1,:] = a[:,0,:] # enforce periodic boundary condions
            if flatten:
                return a.flatten()
        if axis == 2:
            if f is None:
                if higher: # central approximation of a dervative
                    a[:,:,:-1] = (np.roll(a[:,:,:-1], -1, axis = 2) - np.roll(a[:,:,:-1], 1, axis = 2))/2
                else:
                    a[:,:,:-1] = a[:,:,:-1] - np.roll(a[:,:,:-1], 1, axis = 2)
            else:
                # if a finction f given output d f(vec) / dx instead of d(vec)/dx
                if higher:
                    a[:,:,:-1] = (self.vectorized(f, np.roll(a[:,:,:-1], -1, axis = 2)) - self.vectorized(f, np.roll(a[:,:,:-1], 1, axis = 2)))/2
                else:
                    a[:,:,:-1] = self.vectorized(f, a[:,:,:-1]) - self.vectorized(f, np.roll(a[:,:,:-1], 1, axis = 2))
            a[:,:,-1] = a[:,:,0] # enforce periodic boundary condions
            if flatten:
                return a.flatten()
        return a

    def save(self, Solution, name):
        # create a multilevel dataframe and save as a csv file
        columns = pd.MultiIndex.from_product([['S'+str(i+1) for i in range(Solution.shape[0])], self.X])
        solution = pd.Dataframe(Solution, index = self.T, columns = columns)
        solution.to_csv(name)
        
    def upload(self, name):
        Data = pd.read_csv(name, index_col = 0, header = [0,1])
        Data.columns = pd.MultiIndex.from_product([['S'+str(i+1) for i in range(Data.columns.levshape[0])], np.asarray(Data['S1'].columns, dtype = np.float16)])
        
        return Data