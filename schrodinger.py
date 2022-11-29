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
    def __init__(self, x0, u0, v0, tb, X_f, layers, lb, ub):
        
        ########################################
        # Initialize model Inputs and parameters
        
        X0 = torch.concat((x0, 0*x0), 1)
        X_lb = torch.concat((0*tb + lb[0], tb), 1)
        X_ub = torch.concat((0*tb + ub[0], tb), 1)
        
        # Periodic Boundary Conditions
        self.lb = lb
        self.ub = ub
        
        self.x0 = X0[:, 0:1].requires_grad_()
        self.t0 = X0[:, 1:2].requires_grad_()
        
        self.x_lb = X_lb[:, 0:1].requires_grad_()
        self.t_lb = X_lb[:, 1:2].requires_grad_()
        
        self.x_ub = X_ub[:, 0:1].requires_grad_()
        self.t_ub = X_ub[:, 1:2].requires_grad_()
        
        self.x_f = X_f[:, 0:1].float().requires_grad_()
        self.t_f = X_f[:, 1:2].float().requires_grad_()
        
        self.u0 = u0
        self.v0 = v0
        
        self.pinn = PINN(*layers)
        
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
        X = torch.concat([x,t], 1)
        X = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        uv = self.pinn(X)
        u = uv[:, 0:1]
        v = uv[:, 1:2]
        
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u)
                                  , retain_graph = True, create_graph = True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(u)
                                  , retain_graph=True, create_graph = True)[0]
        
        return u, v, u_x, v_x
        
    def net_f_uv(self, x, t):
        u, v, u_x, v_x = self.net_uv(x, t)
        
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u)
                                  , retain_graph = True, create_graph = True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x)
                                  , retain_graph = True, create_graph = True)[0]
        
        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v)
                                  , retain_graph = True, create_graph = True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x)
                                  , retain_graph = True, create_graph = True)[0]
        
        f_u = u_t + 0.5*u_xx + (u**2 + v**2)*v
        f_v = v_t - 0.5*v_xx - (u**2 + v**2)*u
        
        return f_u, f_v
        
    def loss_func(self):
        u0_pred, v0_pred, _, _ = self.net_uv(self.x0, self.t0)
        u_lb_pred, v_lb_pred, u_x_lb_pred, v_x_lb_pred = self.net_uv(self.x_lb, self.t_lb)
        u_ub_pred, v_ub_pred, u_x_ub_pred, v_x_ub_pred= self.net_uv(self.x_ub, self.t_ub) 
        f_u_pred, f_v_pred = self.net_f_uv(self.x_f, self.t_f)       
        
        loss = (torch.mean((self.u0 - u0_pred)**2) +
                     torch.mean((self.v0 - v0_pred)**2) + 
                     torch.mean((u_lb_pred- u_ub_pred)**2) + 
                     torch.mean((v_lb_pred- v_ub_pred)**2) +
                     torch.mean((u_x_lb_pred - u_x_ub_pred)**2) + 
                     torch.mean((v_x_lb_pred - v_x_ub_pred)**2) + 
                     torch.mean((f_u_pred)**2) + torch.mean((f_v_pred)**2)
                     )
        self.optimizer.zero_grad()
        loss.backward()
        
        self.iter += 1
        if self.iter % 100 == 0:
            print(
                "Iter: %d, Loss: %.5f" %
                (
                    self.iter,  loss.item()
            ))
        return loss
    
    def train(self, epochs):
        for epoch in range(epochs):
            self.optimizer_Adam.step(self.loss_func)
        self.optimizer.step(self.loss_func)
    
    
    
        
noise = 0

# Domain bounds
lb = torch.tensor([-5.0, 0.0]).float()
ub = torch.tensor([5.0, np.pi/2]).float()

N0 = 50
N_b = 50
N_f = 20000
layers = [2, 100, 2, 4]

data = scipy.io.loadmat("data/NLS.mat")

t = torch.tensor(data["tt"]).flatten().float()#[:, None]
x = torch.tensor(data["x"]).flatten().float()#[:, None]

Exact = torch.tensor(data["uu"])
Exact_u = torch.real(Exact).float()
Exact_v = torch.imag(Exact).float()
Exact_h = torch.sqrt(Exact_u**2 + Exact_v**2)


X, T = torch.meshgrid(x, t, indexing = 'xy')#.reshape(2, -1) # 1D inputs required

X_star = torch.hstack((X.flatten()[:, None], T.flatten()[:, None]))
u_star = Exact_u.T.flatten()[:, None]
v_star = Exact_v.T.flatten()[:, None]
h_star = Exact_h.T.flatten()[:, None]


# Initial values
idx_x = np.random.choice(x.shape[0], N0, replace = False)
x0 = x[idx_x][:, None]
u0 = Exact_u[idx_x, 0:1]
v0 = Exact_v[idx_x, 0:1]

idx_t = np.random.choice(t.shape[0], N_b, replace = False)
tb = t[idx_t][:, None]

X_f = lb + (ub-lb)*lhs(2, N_f)  # pyDOE lhs used for generating quasi-random multidimensional samples

model = ModelNetwork(x0, u0, v0, tb, X_f, layers, lb, ub)
start = time.time()
model.train(1000)
print("Training took: " + str(time.time() - start))
