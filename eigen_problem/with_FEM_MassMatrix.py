import numpy as np
import scipy.io as sio
import scipy.sparse.linalg as ln
from gridlod.world import World
from gridlod import util, fem, lod, interp, world
import sys
sys.path.insert(0, '/home/kolombag/Documents/gridlod-random-perturbations/random_perturbations')
import build_coefficient, lod_periodic

NFine = np.array([9])    # Number of "fine-blocks" on τ_h mesh in each direction (1-D array: [x_h, y_h, z_h])
NpFine = np.prod(NFine+1)     # Number of "fine-nodes" on τ_h mesh in each direction (1-D array: [x_h, y_h, z_h])
Nepsilon = np.array([6])    # Number of "epsilon-blocks" on τ_ε mesh in each direction (1-D array: [x_ε, y_ε, z_ε] if 3D etc.)
NCoarse = np.array([3])     # Number of "coarse-blocks" on τ_H mesh in each direction (1-D array: [x_H, y_H, z_H]) # Number of layers in the patch 
NpCoarse = np.prod(NCoarse+1) 
k=1                     
NSamples = 1
dim = np.size(NFine)
  
boundaryConditions = None
alpha = 0.5
beta = 1.
#p= 0.1
pList = [0.1] #, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
percentage_comp = 0.15
model ={'name': 'check', 'alpha': alpha, 'beta': beta}
np.random.seed(123)

NCoarseElement = NFine // NCoarse
world = World(NCoarse, NCoarseElement, boundaryConditions)

xpFine = util.pCoordinates(NFine)

#def compute_eigen_vals(NFine, Nepsilon, NCoarse, k, NSamples, world, pList):

NpFine = np.prod(NFine+1) 

def computeKmsij(TInd, a, IPatch):
    patch = lod_periodic.PatchPeriodic(world, k, TInd)
    aPatch = lod_periodic.localizeCoefficient(patch,a, periodic=True)

    correctorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)
    csi = lod.computeBasisCoarseQuantities(patch, correctorsList, aPatch)
    return patch, correctorsList, csi.Kmsij, csi

# LOD for deterministic coeffcient - no updates
basis = fem.assembleProlongationMatrix(world.NWorldCoarse, world.NCoarseElement)
#MFull = fem.assemblePatchMatrix(world.NWorldFine, world.MLocFine)
#MgradFull = fem.assemblePatchMatrix(world.NWorldFine, world.ALocFine)
computePatch = lambda TInd: lod_periodic.PatchPeriodic(world, k, TInd)
patchT = list(map(computePatch, range(world.NtCoarse)))

for p in pList:
    #for N in range(NSamples):
        aPert = build_coefficient.build_randomcheckerboard(Nepsilon,NFine,alpha,beta,p)
        #print('CoeffListSize:',aPert.size)
        #print('CoeffList: \n', aPert)

        #true LOD
        if dim == 2:
            middle = world.NWorldCoarse[1] // 2 * world.NWorldCoarse[0] + world.NWorldCoarse[0] // 2
        else: 
            middle = NCoarse[0] // 2
            
        patchRef = lod_periodic.PatchPeriodic(world, k, middle)
        IPatch = lambda: interp.L2ProjectionPatchMatrix(patchRef)
        computeKmsijT = lambda TInd: computeKmsij(TInd, aPert, IPatch)     

        patchT, correctorsTtrue, KmsijTtrue, _ = zip(*map(computeKmsijT, range(world.NtCoarse)))
        #print('KTtrue: \n', KmsijTtrue)
        #print('KmsijTtrue:', len(KmsijTtrue))
        KFulltrue = lod_periodic.assembleMsStiffnessMatrix(world, patchT, KmsijTtrue, periodic=True)
        #print('Kshape:', KFulltrue.shape)
        #print('Ksize:', KFulltrue.size)
        print('K: \n', KFulltrue)
        correctorsTtrue = tuple(correctorsTtrue)
        modbasistrue = basis - lod_periodic.assembleBasisCorrectors(world, patchT, correctorsTtrue, periodic=True)

        # FEM Mass Matrix
        MFEM = fem.assemblePatchMatrix(world.NWorldCoarse, world.MLocCoarse) 
        #print('FEM1 \n', MFEM)
        #print(MFEM.size)
        #print('M: \n', MFEM.shape)

        if dim == 2:
                KFulltrue.tolil()
                KFulltrue[np.arange(0, NCoarse[1]*(NCoarse[0]+1)+1, NCoarse[0]+1),:] \
                        += KFulltrue[np.arange(NCoarse[0], np.prod(NCoarse+1), NCoarse[0]+1),:]         # Wrap the LHS-RHS-row values together at LHS-mesh boundary points
                KFulltrue[:, np.arange(0, NCoarse[1] * (NCoarse[0] + 1) + 1, NCoarse[0] + 1)] \
                        += KFulltrue[:, np.arange(NCoarse[0], np.prod(NCoarse + 1), NCoarse[0] + 1)]    # Wrap the LHS-RHS-column values together at LHS-mesh boundary points
                KFulltrue[np.arange(NCoarse[0]+1), :] += KFulltrue[np.arange(NCoarse[1]*(NCoarse[0]+1), np.prod(NCoarse+1)), :]          # Wrap the Bottom - top row-values together at BOTTOM-mesh boundary points 
                KFulltrue[:, np.arange(NCoarse[0] + 1)] += KFulltrue[:, np.arange(NCoarse[1] * (NCoarse[0] + 1), np.prod(NCoarse + 1))]  # Wrap the Bottom - top column-values together at BOTTOM-mesh boundary points
                KFulltrue.tocsc()

                fixed_DoF = np.concatenate((np.arange(NCoarse[1] * (NCoarse[0] + 1), NpCoarse), 
                                                np.arange(NCoarse[0], NpCoarse - 1, NCoarse[0] + 1)))    # All the abandoning boundary points
                #print('fixed_size:',fixed_DoF.size)
                #print('fixed: \n', fixed_DoF)
                #print('fixed_shape:', fixed_DoF.shape)
                free_DoF = np.setdiff1d(np.arange(NpCoarse), fixed_DoF)  # Rest of the nodal indices 
                KLOD_Free_DoF = KFulltrue[free_DoF][:, free_DoF]         # Array after BC applied

                MFEM.tolil()
                #print('Mlil: \n', MFEM.tolil())
                MFEM[np.arange(0, NCoarse[1]*(NCoarse[0]+1)+1, NCoarse[0]+1),:] \
                        += MFEM[np.arange(NCoarse[0], np.prod(NCoarse+1), NCoarse[0]+1),:]
                #print('M2: \n', MFEM.shape)
                MFEM[:, np.arange(0, NCoarse[1] * (NCoarse[0] + 1) + 1, NCoarse[0] + 1)] \
                        += MFEM[:, np.arange(NCoarse[0], np.prod(NCoarse + 1), NCoarse[0] + 1)]
                MFEM[np.arange(NCoarse[0]+1), :] += MFEM[np.arange(NCoarse[1]*(NCoarse[0]+1), np.prod(NCoarse+1)), :]
                MFEM[:, np.arange(NCoarse[0] + 1)] += MFEM[:, np.arange(NCoarse[1] * (NCoarse[0] + 1), np.prod(NCoarse + 1))]
                #print('Mshape:', MFEM.shape)
                MFEM.tocsc()
                #print(MFEM.size)
                #print('M: \n', MFEM.tocsc)
                MFEM_Free_DoF = MFEM[free_DoF][:, free_DoF]
        else:
                KFulltrue.tolil()
                KFulltrue[0] += KFulltrue[-1]
                KFulltrue.tocsc() 

                free_DoF = np.setdiff1d(np.arange(NpCoarse-1), KFulltrue[-1])
                KLOD_Free_DoF = KFulltrue[free_DoF][:, free_DoF]

                MFEM.tolil()
                MFEM[0] += MFEM[-1]
                MFEM.tocsc() 

                free_DoF = np.setdiff1d(np.arange(NpCoarse-1), MFEM[-1])
                MFEM_Free_DoF = MFEM[free_DoF][:, free_DoF]
        # Compute for eigen values
        #evals= ln.eigsh(KFulltrue.tocsc(), 1, MFEM.tocsc())
        evals= ln.eigsh(KLOD_Free_DoF , 1,  MFEM_Free_DoF)
        print('Eigen Values: \n', evals[0])
        print('Eigen Vectors:\n', evals[1])



