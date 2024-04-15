import numpy as np
import scipy.io as sio
import time
from offline_online_alg import *
from convergence import *
from with_FEM_MassMatrix import *
from plots import *
from Experimental_Order_Cvg import *

NFine = np.array([256, 256])
Nepsilon = np.array([128,128])
NCoarse = np.array([4,4])
k=3
NSamples = 200
dim = np.size(NFine)
Neigen = 3
boundaryConditions = None
alpha = 0.1
beta = 1.
pList = [0.02, 0.04, 0.06, 0.08, 0.1]
np.random.seed(1)
model ={'name': 'check', 'alpha': alpha, 'beta': beta}
root = 'All_Data/OO_2D_randcheck/N200_Ppt02_pt1_5_K3/'

#convergence(Neigen, NCoarse, NFine, Nepsilon, k, NSamples, pList,alpha,beta, model, solver = "KOOLOD", reference_solver="FEM", save_files = True, plot=False, root=root)

plots_cvg(root=root, H_Convergence=True, p_Convergence=True)