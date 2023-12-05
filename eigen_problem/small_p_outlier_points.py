from with_FEM_MassMatrix import *
from convergence import *
from Reference_Solvers import *
from Experimental_Order_Cvg import *
from plots import *
import numpy as np
import matplotlib.pyplot as plt

alpha = 0.1
beta = 1.
NSamples =200
pList = [0.01, 0.02, 0.03, 0.04,0.05]
model ={'name': 'check', 'alpha': alpha, 'beta': beta}
NCoarse = np.array([4])
Nepsilon = np.array([128])
NFine = np.array([256])    # Number of "epsilon-blocks" on τ_ε mesh in each direction (1-D array: [x_ε, y_ε, z_ε] if 3D etc.)     # Number of "coarse-blocks" on τ_H mesh in each direction (1-D array: [x_H, y_H, z_H]) # Number of layers in the patch 
Neigen = 3
k =3
np.random.seed(123)
root = 'Data/LOD_1d_randcheck/N200_Ppt01_pt1_5_K3/'

def scatterplot_outliers(root=root):
    err = sio.loadmat(root + '_meanErr_H' + str(32) + '.mat')
    Error_λ1 = err['absErr_1']
    Error_λ2 = err['absErr_2']
    pList = err['pList'][0]
    Error_l1_p0 = Error_λ1[0][:]
    Error_l1_p1 = Error_λ1[1][:]
    Error_l2_p0 = Error_λ2[0][:]
    Error_l2_p1 = Error_λ2[1][:]
    
    x = np.linspace(1, 200, num=200)
    plt.scatter(x, Error_l1_p0)
    plt.scatter(x, Error_l2_p0)
    plt.show()


scatterplot_outliers(root=root)