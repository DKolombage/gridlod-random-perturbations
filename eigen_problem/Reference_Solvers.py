import numpy as np
import scipy.io as sio
import scipy.sparse.linalg as ln
from gridlod.world import World
from gridlod import util, fem, lod, interp, world
import sys
sys.path.insert(0, '/home/kolombag/Documents/gridlod-random-perturbations/random_perturbations')
import build_coefficient, lod_periodic

alpha = 0.1
beta = 1.
NSamples = 1
pList = [0.9]
model ={'name': 'check', 'alpha': alpha, 'beta': beta}
NFine = np.array([256])    # NFigure_umber of "fine-blocks" on τ_h mesh in each direction (1-D array: [x_h, y_h, z_h])
Nepsilon = np.array([128])    # Number of "epsilon-blocks" on τ_ε mesh in each direction (1-D array: [x_ε, y_ε, z_ε] if 3D etc.)     # Number of "coarse-blocks" on τ_H mesh in each direction (1-D array: [x_H, y_H, z_H]) # Number of layers in the patch 
#Neigen = 3
#NCoarse = np.array([64])
np.random.seed(123)

def FEM_EigenSolver(Neigen, NCoarse, NFine=NFine,Nepsilon=Nepsilon, NSamples=NSamples, pList=pList):
        NpFine = np.prod(NFine+1)     # Number of "fine-nodes" on τ_h mesh in each direction (1-D array: [x_h, y_h, z_h])
        NpCoarse = np.prod(NCoarse+1) 
        dim = np.size(NFine)
        
        boundaryConditions = None
        percentage_comp = 0.15
        np.random.seed(123)

        NCoarseElement = NFine // NCoarse
        world = World(NCoarse, NCoarseElement, boundaryConditions)
        xpFine = util.pCoordinates(NFine)

        NpFine = np.prod(NFine+1) 

        for p in pList:
                for N in range(NSamples):
                    aPert = build_coefficient.build_randomcheckerboard(Nepsilon,NFine,alpha,beta,p)
                    MFEM = fem.assemblePatchMatrix(world.NWorldFine, world.MLocFine) 
                    KFEM = fem.assemblePatchMatrix(world.NWorldFine, world.ALocFine, aPert) 
                    #print('M \n', MFEM)
                    #print('K \n', KFEM)
                    #print('coeff \n', aPert)

                    if dim == 2:
                            KFEM.tolil()
                            KFEM[np.arange(0, NFine[1]*(NFine[0]+1)+1, NFine[0]+1),:] \
                                    += KFEM[np.arange(NFine[0], np.prod(NFine+1), NFine[0]+1),:]         # Wrap the LHS-RHS-row values together at LHS-mesh boundary points
                            KFEM[:, np.arange(0, NFine[1] * (NFine[0] + 1) + 1, NFine[0] + 1)] \
                                    += KFEM[:, np.arange(NFine[0], np.prod(NFine + 1), NFine[0] + 1)]    # Wrap the LHS-RHS-column values together at LHS-mesh boundary points
                            KFEM[np.arange(NFine[0]+1), :] += KFEM[np.arange(NFine[1]*(NFine[0]+1), np.prod(NFine+1)), :]          # Wrap the Bottom - top row-values together at BOTTOM-mesh boundary points 
                            KFEM[:, np.arange(NFine[0] + 1)] += KFEM[:, np.arange(NFine[1] * (NFine[0] + 1), np.prod(NFine + 1))]  # Wrap the Bottom - top column-values together at BOTTOM-mesh boundary points
                            KFEM.tocsc()

                            fixed_DoF = np.concatenate((np.arange(NFine[1] * (NFine[0] + 1), NpFine), 
                                                            np.arange(NFine[0], NpFine - 1, NFine[0] + 1)))    # All the abandoning boundary points
                            #print('fixed_size:',fixed_DoF.size)
                            #print('fixed: \n', fixed_DoF)
                            #print('fixed_shape:', fixed_DoF.shape)
                            free_DoF = np.setdiff1d(np.arange(NpFine), fixed_DoF)  # Rest of the nodal indices 
                            KFEM_Free_DoF = KFEM[free_DoF][:, free_DoF]         # Array after BC applied
                            #print('KLOD_Free_DoF:', KFEM.shape)

                            MFEM.tolil()
                            #print('Mlil: \n', MFEM.tolil())
                            MFEM[np.arange(0, NFine[1]*(NFine[0]+1)+1, NFine[0]+1),:] \
                                    += MFEM[np.arange(NFine[0], np.prod(NFine+1), NFine[0]+1),:]
                            #print('M2: \n', MFEM.shape)
                            MFEM[:, np.arange(0, NFine[1] * (NFine[0] + 1) + 1, NFine[0] + 1)] \
                                    += MFEM[:, np.arange(NFine[0], np.prod(NFine + 1), NFine[0] + 1)]
                            MFEM[np.arange(NFine[0]+1), :] += MFEM[np.arange(NFine[1]*(NFine[0]+1), np.prod(NFine+1)), :]
                            MFEM[:, np.arange(NFine[0] + 1)] += MFEM[:, np.arange(NFine[1] * (NFine[0] + 1), np.prod(NFine + 1))]
                            #print('Mshape:', MFEM.shape)
                            MFEM.tocsc()
                            #print(MFEM.size)
                            #print('M: \n', MFEM.tocsc)
                            MFEM_Free_DoF = MFEM[free_DoF][:, free_DoF]
                    else:
                            KFEM.tolil()
                            #KFEM[0] += KFEM[-1]
                            KFEM[np.arange(NFine[0], np.prod(NFine+1), NFine[0]+1),:] += KFEM[np.array([0])]
                            KFEM[:, np.array([0])] += KFEM[:, np.arange(NFine[0], np.prod(NFine + 1), NFine[0] + 1)]
                            #print('testKFEM', KFEM)
                            KFEM.tocsc() 

                            fixed_DoF = np.arange(NFine[0], np.prod(NFine + 1), NFine[0] + 1)
                            free_DoF = np.setdiff1d(np.arange(NpFine), fixed_DoF)
                            #free_DoF = np.setdiff1d(np.arange(NpFine-1), KFEM[-1])
                            #KFEM_Free_DoF = KFEM[free_DoF][:, free_DoF]
                            KFEM_Free_DoF = KFEM[1:,:][:,free_DoF]
                            #print('KFEM_Free_DoF:', KFEM_Free_DoF.shape)
                            #print('KFEM_:', KFEM_Free_DoF)

                            MFEM.tolil()
                            MFEM[np.arange(NFine[0], np.prod(NFine+1), NFine[0]+1),:] += MFEM[np.array([0])]
                            MFEM[:, np.array([0])] += MFEM[:, np.arange(NFine[0], np.prod(NFine + 1), NFine[0] + 1)]
                            #MFEM[0] += MFEM[-1]
                            MFEM.tocsc() 

                            fixed_DoF = np.arange(NFine[0], np.prod(NFine + 1), NFine[0] + 1)
                            free_DoF = np.setdiff1d(np.arange(NpFine-1), fixed_DoF)
                            MFEM_Free_DoF = MFEM[1:,:][:,free_DoF]
                            #print('FEM_Free_DoF:',MFEM_Free_DoF.shape)

                    # Compute for eigen values
        evalsFEM= ln.eigsh(KFEM_Free_DoF , Neigen,  MFEM_Free_DoF, sigma =0.05, which='LM', return_eigenvectors = False, tol=1E-2) # v0, (Stiff_Matrix, Number of e.values needed, Mass_Matrix), 
        return evalsFEM[1::]

#B = FEM_EigenSolver(Neigen=Neigen, NCoarse=NCoarse, NFine=NFine, Nepsilon=Nepsilon, NSamples=NSamples, pList=pList)
#print('FEM:\n ',B)