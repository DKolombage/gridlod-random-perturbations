from with_FEM_MassMatrix import *
from convergence import *
from Reference_Solvers import *
from Experimental_Order_Cvg import *
from plots import *

alpha = 0.1
beta = 1.
NSamples =200
pList = pList = [0.02, 0.04, 0.06, 0.08,0.1]
model ={'name': 'check', 'alpha': alpha, 'beta': beta}
NCoarse = np.array([4,4])
NFine = np.array([256, 256])   # Number of "fine-blocks" on τ_h mesh in each direction (1-D array: [x_h, y_h, z_h])
Nepsilon = np.array([128,128])    # Number of "epsilon-blocks" on τ_ε mesh in each direction (1-D array: [x_ε, y_ε, z_ε] if 3D etc.)     # Number of "coarse-blocks" on τ_H mesh in each direction (1-D array: [x_H, y_H, z_H]) # Number of layers in the patch 
Neigen = 3
k =3
np.random.seed(1)
root = 'All_Data/LOD_2D_randcheck/N200_Ppt02_pt1_5_K3/'
type = "rand_checkerboard"

convergence(Neigen, NCoarse, NFine, Nepsilon, k, NSamples, pList,alpha,beta, model, solver = "LOD", reference_solver="FEM", save_files = True, plot=False, root=root)

#plots_cvg(root, H_Convergence=True, p_Convergence=True)