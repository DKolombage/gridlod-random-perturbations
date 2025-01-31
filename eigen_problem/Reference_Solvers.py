import numpy as np
import scipy.io as sio
import scipy.sparse.linalg as ln
from gridlod.world import World
from gridlod import util, fem, lod, interp, world
import sys
sys.path.insert(0, '/home/kolombag/Documents/gridlod-random-perturbations/random_perturbations')
import build_coefficient, lod_periodic
import math

# 1D Exact eigenvalue solver for constant coefficient eigenvalue problem
def Exact_EigenSolver(Neigen):
    eigenvalues = []
    for i in range(Neigen+1):
        if (i%2)==0:
            eig = (i**2)*math.pi**2
        else: 
            eig = ((i-1)**2)*math.pi**2
        eigenvalues.append(eig)
    return eigenvalues[2::] #return  print('Exact Eigenvalues:\n', eigenvalues[1::])        


# FEM solver 1D and 2D
def FEM_EigenSolver(Neigen, NSamples, pList,alpha,beta, NCoarse, NFine, Nepsilon, model, save_file=True): 
        NpFine = np.prod(NFine+1)     # Number of "fine-nodes" on τ_h mesh in each direction (1-D array: [x_h, y_h, z_h])
        NpCoarse = np.prod(NCoarse+1) 
        dim = np.size(NFine)
        
        boundaryConditions = None
        percentage_comp = 0.15
        np.random.seed(1)

        NCoarseElement = NFine // NCoarse
        world = World(NCoarse, NCoarseElement, boundaryConditions)
        xpFine = util.pCoordinates(NFine)

        NpFine = np.prod(NFine+1) 

        FEM_λ1 = np.zeros((len(pList), NSamples))
        FEM_λ2 = np.zeros((len(pList), NSamples))

        for ii in range(len(pList)):
            p= pList[ii]
            for N in range(NSamples):
                if model['name'] == 'check':
                      aPert = build_coefficient.build_randomcheckerboard(Nepsilon,NFine,alpha,beta,p)
                elif model['name'] == 'incl':
                      left = model['left']
                      right = model['right']
                      aPert = build_coefficient.build_inclusions_defect_2d(NFine,Nepsilon,alpha,beta,left,right,p)
                elif model['name'] == 'inclvalue':
                      left = model['left']
                      right = model['right']
                      value = model['defval']
                      aPert = build_coefficient.build_inclusions_defect_2d(NFine,Nepsilon,alpha,beta,left,right,p, value)
                elif model['name'] in ['inclfill', 'inclshift', 'inclLshape']:
                      aPert = build_coefficient.build_inclusions_change_2d(NFine,Nepsilon,alpha,beta,left,right,p,model)
                else:
                      NotImplementedError('other models not available!')
                      
                MFEM = fem.assemblePatchMatrix(world.NWorldFine, world.MLocFine) 
                KFEM = fem.assemblePatchMatrix(world.NWorldFine, world.ALocFine, aPert) 
     

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
                        
                        free_DoF = np.setdiff1d(np.arange(NpFine), fixed_DoF)  # Rest of the nodal indices 
                        KFEM_Free_DoF = KFEM[free_DoF][:, free_DoF]         # Array after BC applied

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
                FEM_λ1[ii, N] = evalsFEM[1]
                FEM_λ2[ii, N] = evalsFEM[2]
        if save_file:
                sio.savemat('FEM_Eigenvalues' + '.mat', {'FEM_1st_Evalue': FEM_λ1, 'FEM_2nd_Evalue': FEM_λ2, 'pList': pList})
        return FEM_λ1, FEM_λ2