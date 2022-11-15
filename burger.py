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


class ModelNetwork:
    def __init__(self) -> None:

        self.model = PINN(input_size=2, hidden_size=20,
                          output_size=1, n_layers=4)

        self.h = 0.1
        self.k = 0.1
        x = torch.arange(-1, 1 + self.h, self.h)
        t = torch.arange(0, 1 + self.k, self.k)

        # exact solution
        self.X = torch.stack(torch.meshgrid(x, t, indexing = 'ij')).reshape(2, -1).T

        # training data such that it fits Burger's equation along with Dirichlet boundary conditions
        # https://maziarraissi.github.io/PINNs/#:~:text=Example%20(Burgers%E2%80%99%20Equation)
        
        bc1 = torch.stack(torch.meshgrid(x[0], t, indexing = 'ij')).reshape(2, -1).T
        bc2 = torch.stack(torch.meshgrid(x[-1], t, indexing = 'ij')).reshape(2, -1).T
        ic = torch.stack(torch.meshgrid(x, t[0], indexing = 'ij')).reshape(2, -1).T
        self.X_train = torch.cat([bc1, bc2, ic])
        
        y_bc1 = torch.zeros(len(bc1))
        y_bc2 = torch.zeros(len(bc2))
        y_ic = -torch.sin(math.pi * ic[:, 0])
        self.y_train = torch.cat([y_bc1, y_bc2, y_ic])
        self.y_train = self.y_train.unsqueeze(1)
        
        self.X.requires_grad = True
        
        # Defime training parameters
        self.criterion = torch.nn.MSELoss()
        self.iter = 1
        
        self.optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            lr = 0.1,
            max_iter = 5e4,
            max_eval= 5e4,
            history_size=50,
            tolerance_change=1.0 * np.finfo(float).eps,
            tolerance_grad=1e-5
        )
        
        self.adam = torch.optim.Adam(self.model.parameters())
        
    def loss_func(self): 
        """
        Define loss functions describing physics of the system
        """
        # https://maziarraissi.github.io/PINNs/#:~:text=The%20shared%20parameters%20between,mean%20squared%20error%20loss
        
        self.optimizer.zero_grad()
        y_pred = self.model(self.X_train)
        loss_data = self.criterion(y_pred, self.y_train)
        
        u = self.model(self.X)
        
        du_dX = torch.autograd.grad(inputs =self.X, outputs = u,grad_outputs=  torch.ones_like(u),
                                    retain_graph=True, create_graph=True)[0]
        du_dt = du_dX[:, 1]
        du_dx = du_dX[:, 0]
        du_dxx = torch.autograd.grad(inputs =self.X, outputs = du_dX, grad_outputs=  torch.ones_like(du_dX),
                                    retain_graph=True, create_graph=True)[0][:, 0]
        
        loss_pde = self.criterion(du_dt + u.squeeze() * du_dx, 0.01 /math.pi * du_dxx)
        
        loss = loss_pde + loss_data
        loss.backward()
        if self.iter % 100 == 0:
            print(self.iter ,  loss.item())
        self.iter == self.iter + 1
        
        return loss
    
    def train(self, epochs):
        for i in range(epochs):
            self.adam.step(self.loss_func)
            self.optimizer.step(self.loss_func)
            
            

net = ModelNetwork()
net.train(1000)

# Evaluating model

h = 0.01
k = 0.01
x = torch.arange(-1, 1, h)
t = torch.arange(0, 1, k)

# Exact solution = 
X = torch.stack(torch.meshgrid(x, t, indexing = 'ij')).reshape(2, -1).T

model = net.model
model.eval()
with torch.no_grad():
    y_pred = model(X).reshape(len(x), len(t)).numpy()
    
sns.heatmap(y_pred, cmap = "jet")
plt.savefig("heatmap.png")
        
        
