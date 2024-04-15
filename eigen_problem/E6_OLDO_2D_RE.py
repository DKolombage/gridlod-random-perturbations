import numpy as np
import scipy.io as sio
import time
from offline_online_alg import *
from convergence import *
from with_FEM_MassMatrix import *
from plots import *
from Experimental_Order_Cvg import *

NFine = np.array([256,256])
Nepsilon = np.array([128,128])
NCoarse = np.array([4,4])
k=3
NSamples = 200
dim = np.size(NFine)
Neigen = 3
alpha = 0.1
beta = 1
left = np.array([0.25, 0.25])
right = np.array([0.75, 0.75])
pList = [0.02,0.04,0.06,0.08,0.1]
def_values = [alpha, 0.5, 5.]
modelincl = {'name': 'incl', 'bgval': alpha, 'inclval': beta, 'left': left, 'right': right}
np.random.seed(1)
root = 'All_Data/OO_2D_randerasure/N200_Ppt02_pt1_5_K3/'

convergence(Neigen, NCoarse, NFine, Nepsilon, k, NSamples, pList,alpha,beta, modelincl, solver = "KOOLOD", reference_solver="FEM", save_files = True, plot=False, root=root)
plots_cvg(root, H_Convergence=True, p_Convergence=True)