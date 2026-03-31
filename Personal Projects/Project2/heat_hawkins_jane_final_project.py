# Earth 410 Final Exam
# Jane Hawkins

#exec(open("heat_hawkins_jane_final_project.py").read())

# build a PINN to solve 1D heat equation

import torch
from torch import nn
from torch.nn import functional as F
torch.set_default_dtype(torch.float64)
from torch.optim import Adam, SGD
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import pyplot as plt

tf = 0.5
L = 1
kappa = 2
tmax = 0.6
sigma = 0.8
alpha = 0.01   # learning rate

def p(t):
   return 0.25 * torch.cos(2 * torch.pi * torch.tensor(t) / tmax) + 0.5

def pprime(t):
   return -0.25 * 2*torch.pi/tmax * torch.sin(2*torch.pi*t/tmax)

def exact(x, t):
   return torch.exp(-(x - p(t))**2/(2*sigma**2))

def exact_x(x, t):
   return exact(x,t) * -2*(x - p(t))/(2*sigma**2)

def exact_xx(x, t):
   return exact_x(x, t) * -2*(x - p(t))/(2*sigma**2) + exact(x,t)*(-2/(2*sigma**2))

def exact_t(x, t):
   return exact(x,t)   * -2*(x - p(t))/(2*sigma**2)  * -pprime(t)

def s(x, t):
   return exact_t(x,t) - kappa * exact_xx(x,t)

def h(t):
   return exact(0, t)

def g(t):
   return exact(L, t)

def T0(x):
   return exact(x, 0)

def plot_stuff(X, T, T_pred, Te, loss):
    # Create a figure and a 1x2 grid of subplots (1 row, 2 columns)
    # The `axes` variable will be an array of Axes objects, one for each subplot.
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 4)) # Adjust figsize as needed

    # Plot on the first subplot (left)
    axes[0].pcolormesh(X, T, T_pred, cmap='viridis', shading='auto', vmin=0.6, vmax=1)
    axes[0].set_title('Prediction')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('t')

    # Plot on the second subplot (middle)
    axes[1].pcolormesh(X, T, Te, cmap='viridis', shading='auto', vmin=0.6, vmax=1)
    axes[1].set_title('Exact')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('t')

    # Plot on the third subplot showing loss per epoch
    epochs = np.arange(EPOCHS) # array for x-axis epochs
    axes[2].plot(epochs, loss)
    axes[2].set_title('Loss for Each Epoch')
    axes[2].set_xlabel('epoch')
    axes[2].set_ylabel('loss')

    # Adjust layout to prevent titles/labels from overlapping
    plt.tight_layout()

    # Display the plots
    plt.show()


class Neural_Network(nn.Module):
    # sefl-defined class that takes from the nn.Module superclass to build my PINN
    def __init__(self, input_size, hidden_size):
       super().__init__()
       self.linear1 = nn.Linear(input_size, hidden_size, bias=True)
       self.linear2 = nn.Linear(hidden_size, hidden_size, bias=True)
       self.linear3 = nn.Linear(hidden_size, 1, bias=True)
       self.opt = torch.optim.Adam(self.parameters(), lr=alpha)

    def forward(self, Z):
       Z = self.linear1(Z)
       Z = F.tanh(Z)
       Z = self.linear2(Z)
       Z = F.tanh(Z)
       Z = self.linear3(Z)
       return Z
  

# Define the PDE:
def pde_relation(z):
    '''
    z: ordered pair (x, t) that is input to the neural network. 
    '''

    x, t = z.split(1, dim = 1) # unpack z into x and t

    # use autograd to get first-order network derivatives
    grad, = torch.autograd.grad(NN(z).sum(), z, create_graph=True) # compute the derivative
    dT_dx, dT_dt = grad.split(1, dim = 1) # unpack gradient into first order partial derivatives

    # compute the derivative (again) to get second derivative:
    grad_dx, = torch.autograd.grad(dT_dx.sum(), z, create_graph=True) 
    d2T_dxx, d2T_dxt = grad_dx.split(1, dim = 1) # unpack to get partial derivatives

    return dT_dt - kappa * d2T_dxx - s(x, t)

# Define a function to compute the total loss:
def total_loss(z_int, z_initial, z_LB, z_RB):
    '''
    z_int: interior collocation points (ordered pairs x, t)
    z_initial: collocation points at t = 0 boundary 
    z_LB: collocation points at x = 0 (left boundary)
    z_RB: collocation points at x = L (right boundary)
    '''

    #x, t = z_int.split(1, dim = 1)
    x0, t0 = z_initial.split(1, dim = 1) # get x values along t = 0 boundary
    xLB, tLB = z_LB.split(1, dim = 1)    # get t values along x = 0 boundary
    xRB, tRB = z_RB.split(1, dim = 1)    # get t values along x = L boundary

    # compute loss from the PDE (mean squared error):
    loss_PDE = pde_relation(z_int).abs().square().sum() / z_int.shape[0]
    
    # compute loss from left boundary condition:
    loss_LB = (NN(z_LB) - h(tLB)).abs().square().sum() / z_LB.shape[0] 

    # compute loss from right boundary condition:
    loss_RB = (NN(z_RB) - g(tRB)).abs().square().sum() / z_RB.shape[0] # Tnn(L, ti) - G
    
    # compute loss from initial condition:
    loss_initial = (NN(z_initial) - T0(x0)).abs().square().sum() / z_initial.shape[0] # Tnn(xi, 0) - u0(xi)

    # sum to get total loss:
    total = loss_PDE + loss_LB + loss_RB + loss_initial # add up all losses
    
    return total, loss_PDE, loss_LB, loss_RB, loss_initial

#######  GENERATE COLLOCATION POINTS #############

# First generate M collocation points (ordered pairs) within the interior:
M = 20  # there will be M^2 total collocation pionts in interior
x = np.random.uniform(0, L, M)
t = np.random.uniform(0, tf, M)
XX, TT = np.meshgrid(x, t) 
XX = np.reshape(XX, (M*M)) 
TT = np.reshape(TT, (M*M))
XX = torch.tensor(XX, requires_grad=True).unsqueeze(1) # change to tensor
TT = torch.tensor(TT, requires_grad=True).unsqueeze(1) # change to tensor
z_int = torch.cat([XX, TT], dim=1)

m = 100  # Number of collocation points (ordered pairs) at all boundaries

tb = np.random.uniform(0, tf, m) # first generate some random points along t
xb = np.random.uniform(0, L, m) # then generate some random points along x

# Convert tb and xb to Pytorch tensors
tb = torch.tensor(tb, requires_grad=True).unsqueeze(1) # change to m x 1 tensor
xb = torch.tensor(xb, requires_grad=True).unsqueeze(1) # change to m x 1 tensor

# Create the collocation points (ordered pairs) along left and right boundaries:
x0 = torch.tensor(0*torch.ones(m,1), requires_grad=True)# change to M x 1 tensor
x1 = torch.tensor(L*torch.ones(m,1), requires_grad=True)# change to M x 1 tensor
zLB = torch.cat([x0, tb], dim=1) # collocation points along left boundary
zRB = torch.cat([x1, tb], dim=1) # collocation points along right boundary

# Create the collocation points (ordered pairs) at t = 0:
z_initial = torch.cat([xb, torch.zeros(m,1)], dim=1) # collocation points along t = 0 boundary

###### DONE GENERATING COLLOCATION POINTS ########


# CREATE INSTANCE OF NEURAL NETWORK CLASS:
N = 20 # number of neurons in each layer.
NN = Neural_Network(2, N) #input_size = 2, as inputs are order pairs (x, t)


######## BEGIN NETWORK TRAINING #############

EPOCHS = 500   # number of training epochs.

loss_per_epoch = []
for i in range(EPOCHS):

    # Zero your gradients for every batch!
    NN.opt.zero_grad()

    # Compute the loss and its gradients:
    loss, loss_PDE, loss_LB, loss_RB, loss_initial = total_loss(z_int, z_initial, zLB, zRB)
    
    # Append loss to list
    loss_per_epoch.append(loss.item())
    loss.backward()

    # Adjust learning weights
    NN.opt.step()

####### DONE TRAINING #######


#####  START TESTING ########

# create a regular grid on which to test:
x = np.linspace(0, L, 100)
t = np.linspace(0, tf, 100)
X, time = np.meshgrid(x, t)  # create a bunch of ordered pairs (x, t)

# Evaluate exact solution on this grid: 
Texact = exact(torch.tensor(X), torch.tensor(time)).detach().numpy()

# Re-organize grid and send to forward pass:
XX = torch.tensor(X.reshape(10000, 1))
TT = torch.tensor(time.reshape(10000, 1))
Z = torch.cat([XX, TT], dim=1)
Tnn =  NN(Z).detach().numpy()
Tnn = Tnn.reshape(100, 100)

plot_stuff(X, time, Tnn, Texact, loss_per_epoch)

# Review:
# 
# Initially, I was changing around certain variables just to try and get predicted heat map to match the exact solution.
# From this, I found which ones worked best to get the most accurate visual. I used the inital vaiables of the number of 
# epochs = 5000, the optimizer as Adam, and the learning rate of 0.01 as my baseline variables. From here, I wanted to plot 
# the loss per epoch. I found that by reducing the number of epochs from 5000 to 500 gave me a much more gradual decline on my
# plot - which is to be expected when using a smaller sample size - so this wasn't a huge discovery. However, when changing the
# learning rate (from 0.01 to 0.05), keeping all other variables the same, my heat map stayed the same but my loss plot grew some
# wacky points. While it did follow the same general shape, it seemed as though there were more outliers in the loss amount
# for certain epochs. The last variable I tested was the optimizer. Using the optimizer SGD (with learning rate 0.01) it gave
# me the smoothest, most consistent plot from loss per epoch. However, using SGD vs Adam my heat map is much less accurate.
# Overall, I found that Adam maintained its overall accuracy for larger and smaller epoch sizes, whereas SGD kind of blew
# up when I tried to increase the epochs to 5000. Adam also preserved the accuracy of both plots - not favoring one over the other.
# In conclusion, I found that epoch size itself didn't effec the accuracy of the outcome as much as other variables like the
# optimizer and the learning rate, but it did help provide a better looking visual.