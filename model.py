import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class MultiLayerNN(nn.Module):
    def __init__(self, input_dim, 
                h_dim, 
                output_dim, 
                n_layers, 
                residue = True):
        super(MultiLayerNN, self).__init__()
        self.residue = residue
        self.in_fc = nn.Linear(input_dim, h_dim).double()  
        self.hlayer = nn.ModuleList([nn.Linear(h_dim, h_dim).double() for _ in range(n_layers)]) 
        self.out_fc = nn.Linear(h_dim, output_dim).double() 

    def forward(self, x):
        x = torch.relu(self.in_fc(x))
        for i in range(len(self.hlayer)):  
            if self.residue:
                x = torch.relu(self.hlayer[i](x)) + x
            else: 
                x = torch.relu(self.hlayer[i](x))  
        x = self.out_fc(x)
        return x

class BallBot2D(nn.Module):
    def __init__(self, h_dim_x, n_layers_x, h_dim_u, n_layers_u):
        super(BallBot2D, self).__init__()
        # Constants
        self.l = 0.5     #m
        self.Rs = 0.12   #m
        self.Ms = 2.5    #kg
        self.Mb = 4.59      #kg
        self.Js = 0.006  #kgm^2
        self.Jb = 0.2  #kgm^2
        self.J_alpha = 12.48 #kgm^2
        self.J_beta = 12.48  #kgm^2
        self.g = 9.8     #ms^2

        # Assuming D+G is zero for simplicity, however, it is learnable
        #self.sigma = nn.Parameter(torch.rand(2,2))
        self.C = torch.eye(4,4).double()
        #self.u = nn.Parameter(torch.rand(1,1))

        self.FNN_X = MultiLayerNN(input_dim = 4, 
                h_dim = h_dim_x, 
                output_dim = 4, 
                n_layers = n_layers_x, 
                residue = True)
        self.FNN_U = MultiLayerNN(input_dim = 4, 
                h_dim = h_dim_u, 
                output_dim = 1, 
                n_layers = n_layers_u, 
                residue = True)

    def forward(self, x, debug = False):
        x = self.FNN_X.forward(x.flatten())
        u = self.FNN_U.forward(x.flatten())
        M = self.compute_mass_matrix()
        #K_H = self.zero_KH()
        # A-matrix
        state_matrix = self.compute_state_matrix(M)
        # B-matrix
        control_matrix = self.compute_control_matrix(M)
        # COMPUTE X-Dot
        Ax = torch.matmul(state_matrix, x).unsqueeze(1)
        Bu = torch.matmul(control_matrix, u.double())
        Bu = Bu.reshape(4,1)
        dotx = Ax + Bu
        if debug:
            print("A: {}".format(state_matrix))
            print("B: {}".format(control_matrix))
            print("Ax is:{}".format(Ax.shape))
            print("Bu is:{}".format(Bu.shape))
            print('dotX is:{}'.format(dotx.shape))
        return dotx, x

    def compute_mass_matrix(self):
        # Mass inertia matrix M
        M11 = (self.Ms+self.Ms)*self.Rs**2 + self.Js
        M12 = M21 = self.Mb*self.l*self.Rs
        M22 =self.Mb*self.l**2 + self.J_alpha
        M = np.array([[M11, M12],
              [M21, M22]])
        return torch.tensor(M)

    def zero_KH(self):
        # Input matrices
        K_H = np.array([[0, 0],
            [0, -self.Mb*self.g*self.l]])
        return K_H

    def compute_state_matrix(self,M):
        # Determinant of M
        d = np.linalg.det(M)
        # State matrix A
        A = np.array([[0, 0, 1, 0],
              [0, 0, 0, 1],
              [0, (self.Mb*self.g*self.l/d) * M[0, 1], 0, 0],
              [0, (self.Mb*self.g*self.l/d) * M[0, 0], 0, 0]])
        return torch.tensor(A, dtype=torch.float64)
    
    def compute_control_matrix(self,M):
        # Determinant of M
        d = np.linalg.det(M)
        # Input matrix B
        B = np.array([[0],
                    [0],
                    [(M[1, 1] + M[0, 1]) / d],
                    [(M[1, 0] + M[0, 0]) / d]])
        return torch.tensor(B, dtype=torch.float64)
        