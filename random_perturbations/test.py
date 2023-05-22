import numpy as np
import scipy.io as sio
from scipy.linalg import eigh
import build_coefficient
import lod_periodic
from gridlod.world import World
from gridlod import util, fem, lod, interp


NFine = np.array([256, 256])    # Number of "fine-blocks" on τ_h mesh in each direction (1-D array: [x_h, y_h, z_h])
NpFine = np.prod(NFine+1)     # Number of "fine-nodes" on τ_h mesh in each direction (1-D array: [x_h, y_h, z_h])
Nepsilon = np.array([128,128])    # Number of "epsilon-blocks" on τ_ε mesh in each direction (1-D array: [x_ε, y_ε, z_ε] if 3D etc.)
NCoarse = np.array([32,32])     # Number of "coarse-blocks" on τ_H mesh in each direction (1-D array: [x_H, y_H, z_H])
k=4                           # Number of layers in the patch

NSamples = 250
dim = np.size(NFine)

boundaryConditions = None
alpha = 0.1
beta = 1.
pList = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
percentage_comp = 0.15
model ={'name': 'check', 'alpha': alpha, 'beta': beta}
np.random.seed(123)

NCoarseElement = NFine // NCoarse
world = World(NCoarse, NCoarseElement, boundaryConditions)

xpFine = util.pCoordinates(NFine)

def compute_eigen_vals(NFine, Nepsilon, NCoarse, k, NSamples, world, pList):

    NpFine = np.prod(NFine+1) 

    def computeKmsij(TInd, a, IPatch):
        patch = lod_periodic.PatchPeriodic(world, k, TInd)
        aPatch = lod_periodic.localizeCoefficient(patch,a, periodic=True)

        correctorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)
        csi = lod.computeBasisCoarseQuantities(patch, correctorsList, aPatch)
        return patch, correctorsList, csi.Kmsij, csi

    # LOD for deterministic coeffcient - no updates
    basis = fem.assembleProlongationMatrix(world.NWorldCoarse, world.NCoarseElement)
    MFull = fem.assemblePatchMatrix(world.NWorldFine, world.MLocFine)
    MgradFull = fem.assemblePatchMatrix(world.NWorldFine, world.ALocFine)
    computePatch = lambda TInd: lod_periodic.PatchPeriodic(world, k, TInd)
    patchT = list(map(computePatch, range(world.NtCoarse)))

    for p in pList:
        if p == 0.1:
            mean_time_true = 0.
            mean_time_perturbed = 0.

        for N in range(NSamples):
            aPert = build_coefficient.build_randomcheckerboard(Nepsilon,NFine,alpha,beta,p)

            #true LOD
            middle = world.NWorldCoarse[1] // 2 * world.NWorldCoarse[0] + world.NWorldCoarse[0] // 2
            patchRef = lod_periodic.PatchPeriodic(world, k, middle)
            IPatch = lambda: interp.L2ProjectionPatchMatrix(patchRef)
            computeKmsijT = lambda TInd: computeKmsij(TInd, aPert, IPatch)
            if p == 0.1:
                tic = time.perf_counter()
                patchT, correctorsTtrue, KmsijTtrue, _ = zip(*map(computeKmsijT, range(world.NtCoarse)))
                KFulltrue = lod_periodic.assembleMsStiffnessMatrix(world, patchT, KmsijTtrue, periodic=True)
                toc = time.perf_counter()
                mean_time_true += (toc-tic)
                correctorsTtrue = tuple(correctorsTtrue)
                modbasistrue = basis - lod_periodic.assembleBasisCorrectors(world, patchT, correctorsTtrue, periodic=True)
            else:
                patchT, correctorsTtrue, KmsijTtrue, _ = zip(*map(computeKmsijT, range(world.NtCoarse)))
                KFulltrue = lod_periodic.assembleMsStiffnessMatrix(world, patchT, KmsijTtrue, periodic=True)
                correctorsTtrue = tuple(correctorsTtrue)
                modbasistrue = basis - lod_periodic.assembleBasisCorrectors(world, patchT, correctorsTtrue, periodic=True)

        Mass = fem.assemblePatchMatrix(world.NWorldCoarse, world.MLocCoarse) # Fix for Periodic BC

    # Compute for eigen values
        evals, evecs = eigh(KFulltrue, Mass)
    return evals

compute_eigen_vals(NFine, Nepsilon, NCoarse, k, NSamples, world, pList)
# Assemble Mass matrices with Periodic B.C.

# FEM Stiffness and mass matrices
# AFull = fem.assemblePatchMatrix(world.NWorldFine, world.ALocFine, aPert)
# Mass = fem.assemblePatchMatrix(world.NWorldFine, world.MLocFine)
