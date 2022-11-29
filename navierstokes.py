import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyDOE import lhs

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict

import scipy.io
from sklearn.feature_selection import mutual_info_regression

import time
from tqdm import tqdm

sns.set_style("white")


# Schrödinger equation is a linear partial differential equation that 
# governs the wave function of a quantum-mechanical system.
# It is used to predicts the future behavior of a dynamic system.

# Create MLP for model training



class PINN(nn.Module):
    def __init__(self, input_size, hidden_size,
                 output_size, n_layers) -> None:
        super(PINN, self).__init__()

        layers = [("input", torch.nn.Linear(input_size, hidden_size))]
        layers.append(("input_activation", torch.nn.Tanh()))

        # For every other layer
        for i in range(n_layers):
            layers = layers + [
                (f"hidden_{i}", torch.nn.Linear(hidden_size, hidden_size)),
                (f"activation{i}", torch.nn.Tanh()),
            ]

        layers.append(("output", torch.nn.Linear(hidden_size, output_size)))

        layerDict = OrderedDict(layers)
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        ##print(type(x))
        out = self.layers(x)
        return out


class ModelNetwork():
    def __init__(self, x, y, t, u, v, layers):
        
        ########################################
        # Initialize model Inputs and parameters
        
        X = torch.concat((x, y, t), 1)
        self.X = X
        
        # Periodic Boundary Conditions
        self.lb = X.min(0)
        self.ub = X.max(0)
        
        self.x = X[:, 0:1].requires_grad_()
        self.y = X[:, 1:2].requires_grad_()
        self.t = X[:, 2:3].requires_grad_()
        
        self.u = u
        self.v = v
        
        self.lambda_1 = torch.tensor([0.00001], requires_grad=True).float()
        self.lambda_2 = torch.tensor([0.00001], requires_grad=True).float()
        
        self.pinn = PINN(*layers)
        self.pinn.register_parameter("lambda1", self.lambda1)
        self.pinn.register_parameter("lambda2", self.lambda2)
        
        self.optimizer = torch.optim.LBFGS(
            self.pinn.parameters(),
            lr = 1.0, 
            max_iter = 50000,
            max_eval = 50000,
            history_size = 50,
            tolerance_change=1.0 * np.finfo(float).eps
        )
        self.optimizer_Adam = torch.optim.Adam(self.pinn.parameters())
        self.iter = 0
        
    def net_NS(self, x, t):
        
        lambda1 = self.lambda_1
        lambda2 = self.lambda_2
        
        X = torch.concat([x, y, t], 1)
        X = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        psi_and_p = self.pinn(X)
        
        psi = psi_and_p[:, 0:1]
        p = psi_and_p[:, 1:2]
        
        u = torch.autograd.grad(psi, y, grad_outputs=torch.ones_like(psi)
                                  , retain_graph = True, create_graph = True)[0]
        v = -torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi)
                                  , retain_graph=True, create_graph = True)[0]
        
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u)
                                  , retain_graph = True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u)
                                  , retain_graph = True, create_graph = True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u)
                                  , retain_graph = True, create_graph = True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u)
                                  , retain_graph = True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u)
                                  , retain_graph = True)[0]
        
        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(u)
                                  , retain_graph = True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(u)
                                  , retain_graph = True, create_graph = True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(u)
                                  , retain_graph = True, create_graph = True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(u)
                                  , retain_graph = True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(u)
                                  , retain_graph = True)[0]
        
        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(u)
                                  , retain_graph = True)[0]
        p_y = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(u)
                                  , retain_graph = True)[0]
        
        f_u = u_t + lambda1*(u * u_x + v*u_y) + p_x - lambda2*(u_xx + u_yy)
        f_v = v_t + lambda1*(v * v_x + v*v_y) + p_y - lambda2*(v_xx + v_yy)
        
        
        return u, v, p, f_u, f_v

        
    def loss_func(self):
        self.u_pred, self.v_pred, self.p_pred ,self.f_pred, self.f_u_pred, self.f_v_pred = self.net_NS(
            self.x,self.y,self.t
        )    
        
        
        self.loss = (torch.sum((self.u - self.u_pred)**2) +
                     torch.sum((self.v - self.v_pred)**2) + 
                     torch.sum((self.f_u_pred)**2) + 
                     torch.sum((self.f_v_pred)**2) 
                     )
        self.optimizer.zero_grad()
        self.loss.backward()
        
        self.iter += 1
        if self.iter % 100 == 0:
            print(
                "Iter: %d, Loss: %.5f, L1: %.3f, L2: %.5f, Time: %.2f" %
                (
                    self.iter,  self.loss.item(), 
                    self.lambda_1.item(), self.lambda_2.item(), time.time() - self.start
            ))
            self.start = time.time()
        return self.loss
    
    def train(self, epochs):
        self.start = time.time()
        for epoch in range(epochs):
            self.optimizer_Adam.step(self.loss_func)
        self.optimizer.step(self.loss_func)
    

        
    
    
        
N_train = 5000

layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]

# Load Data
data = scipy.io.loadmat('data/cylinder_nektar_wake.mat')
        
U_star = torch.tensor(data['U_star']).float() # N x 2 x T
P_star = torch.tensor(data['p_star']).float() # N x T
t_star = torch.tensor(data['t']).float() # T x 1
X_star = torch.tensor(data['X_star']).float() # N x 2

N = X_star.shape[0]
T = t_star.shape[0]

# Rearrange Data 
XX = torch.tile(X_star[:,0:1], (1,T)) # N x T
YY = torch.tile(X_star[:,1:2], (1,T)) # N x T
TT = torch.tile(t_star, (1,N)).T # N x T

UU = U_star[:,0,:] # N x T
VV = U_star[:,1,:] # N x T
PP = P_star # N x T

x = XX.flatten()[:,None] # NT x 1
y = YY.flatten()[:,None] # NT x 1
t = TT.flatten()[:,None] # NT x 1

u = UU.flatten()[:,None] # NT x 1
v = VV.flatten()[:,None] # NT x 1
p = PP.flatten()[:,None] # NT x 1

######################################################################
######################## Noiseles Data ###############################
######################################################################
# Training Data    
idx = np.random.choice(N*T, N_train, replace=False)
x_train = x[idx,:]
y_train = y[idx,:]
t_train = t[idx,:]
u_train = u[idx,:]
v_train = v[idx,:]

# Training
model = ModelNetwork(x_train, y_train, t_train, u_train, v_train, layers)
model.train(200000)

# Test Data
snap = np.array([100])
x_star = X_star[:,0:1]
y_star = X_star[:,1:2]
t_star = TT[:,snap]

u_star = U_star[:,0,snap]
v_star = U_star[:,1,snap]
p_star = P_star[:,snap]

# Prediction
u_pred, v_pred, p_pred = model.predict(x_star, y_star, t_star)
lambda_1_value = model.sess.run(model.lambda_1)
lambda_2_value = model.sess.run(model.lambda_2)

# Error
error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
error_v = np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2)
error_p = np.linalg.norm(p_star-p_pred,2)/np.linalg.norm(p_star,2)

error_lambda_1 = np.abs(lambda_1_value - 1.0)*100
error_lambda_2 = np.abs(lambda_2_value - 0.01)/0.01 * 100

print('Error u: %e' % (error_u))    
print('Error v: %e' % (error_v))    
print('Error p: %e' % (error_p))    
print('Error l1: %.5f%%' % (error_lambda_1))                             
print('Error l2: %.5f%%' % (error_lambda_2))  
