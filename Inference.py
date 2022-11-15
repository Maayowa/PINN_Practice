import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict

import scipy.io
from sklearn.feature_selection import mutual_info_regression

from tqdm import tqdm

sns.set_style("white")

# Set Seed
seed = 120
# torch.random.seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

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
        out = self.layers(x)
        return out
    


class ModelNetwork():
    # lb --> lower bound; ub --> upper bound
    def __init__(self, X, u, model_params: list, lb, ub):

        ###########################################
        # Define the Inputs and model parameters

        # Boundary conditions
        self.lb = torch.tensor(lb).float()
        self.ub = torch.tensor(ub).float()

        # Data
        self.x = torch.tensor(X[:, 0:1], requires_grad=True).float()
        # Remember X is a stack of x and t
        self.t = torch.tensor(X[:, 1:2], requires_grad=True).float()
        self.u = torch.tensor(u).float()

        # settings
        self.lambda1 = torch.tensor([0.0], requires_grad=True)
        self.lambda2 = torch.tensor([-6.0], requires_grad=True)  # Subject to change this 5.7 used in paper

        self.lambda1 = torch.nn.Parameter(self.lambda1)
        self.lambda2 = torch.nn.Parameter(self.lambda2)

        # Defining Neural network
        self.pinn = PINN(*model_params)
        self.pinn.register_parameter("lambda1", self.lambda1)
        self.pinn.register_parameter("lambda2", self.lambda2)

        # Optimizers
        self.optimizer = torch.optim.LBFGS(
            self.pinn.parameters(),
            lr=1.0,
            max_iter=5e4,
            max_eval=5e4,
            history_size=50,
            tolerance_change=1.0 * np.finfo(float).eps
            # line_search_fn= "strong_wolfe"
        )
        self.optimizer_Adam = torch.optim.Adam(self.pinn.parameters())
        self.iter = 0

        ########################################

    def net_u(self, x, t):
        # Prediction for the latent solution
        u = self.dnn(torch.cat([x, t], dim=1))
        return u


    def net_f(self, x, t):
        """Computing Residuals using Pytorch Autograd
            The main Use of Autograd for partial differentiations for Physics Informed models
            At least according to  my understanding
        """
            
        lambda1 = self.lambda1
        lambda2 = torch.exp(lambda2)
        u = self.net_u(x, t)
        
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                                  retain_graph=True, create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), 
                                  retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs = torch.ones_like(u_x), 
                                   retain_graph=True, create_graph=True)[0]
        
        f = u_t + lambda1 * u * u_x - lambda2 * u_xx
        return f
    
    
    def loss_func(self):
        u_pred = self.net_u(self.x, self.t)
        f_pred = self.net_f(self.x, self.t)
        loss = torch.mean((self.u - u_pred) ** 2) + torch.mean(f_pred **2)
        self.optimizer.zero_grad()
        loss.backward()
        
        self.iter += 1
        if self.iter % 100 == 0:
            print(
                "Loss: %e, Lambda 1: %.5f, Lambda 2: %.5f" %
                (
                    loss.item(),
                    self.lambda1.item(),
                    torch.exp(self.lambda2).item()
                )
            )
        return loss
    
    
    def train(self, epochs):
        self.pinn.train()
        for epoch in range(epochs):
            u_pred = self.net_u(self.x, self.t)
            f_pred = self.net_f(self.x, self.t)
            loss = torch.mean((self.u - u_pred) ** 2) + torch.mean(f_pred ** 2)
            
            # Backward and Optimize
            self.optimizer_Adam.zero_grad()
            loss.backward()
            self.optimizer_Adam.step()
            
            if epoch % 100 == 0:
                print(
                    "Iter: %d, Loss: %.5e, Lambda 1: %.5f, Lambda 2: %.5f" %
                    (
                        epoch,
                        loss.item(),
                        self.lambda1.item(),
                        torch.exp(self.lambda1).item()
                    ))
        self.optimizer.step(self.loss_func)
        
        
    def predict(self, X):
        # Data
        self.x = torch.tensor(X[:, 0:1], requires_grad=True).float()
        # Remember X is a stack of x and t
        self.t = torch.tensor(X[:, 1:2], requires_grad=True).float()
        
        self.pinn.eval()
        u = self.pinn(X)
        t = self.pinn(X)
        u = u.numpy()
        f = f.numpy()
        
        return u, f
            


# https://github.com/jayroxis/PINNs/blob/master/Burgers%20Equation/Burgers%20Identification%20(PyTorch).ipynb

nu = 0.01/np.pi
N_u = 2000
model_params = [2, 20, 1, 7]