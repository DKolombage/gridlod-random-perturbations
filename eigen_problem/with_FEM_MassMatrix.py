import numpy as np
import scipy.io as sio
from scipy.linalg import eigh
from random_perturbations import build_coefficients
from random_perturbations import lod_periodic
from gridlod.world import World


NFine = np.array([16, 16])    # Number of "fine-blocks" on τ_h mesh in each direction (1-D array: [x_h, y_h, z_h])
NpFine = np.prod(NFine+1)     # Number of "fine-nodes" on τ_h mesh in each direction (1-D array: [x_h, y_h, z_h])
Nepsilon = np.array([8,8])    # Number of "epsilon-blocks" on τ_ε mesh in each direction (1-D array: [x_ε, y_ε, z_ε] if 3D etc.)
NCoarse = np.array([4,4])     # Number of "coarse-blocks" on τ_H mesh in each direction (1-D array: [x_H, y_H, z_H])
k=2                           # Number of layers in the patch

NSamples = 1
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

# Assemble Stiffness and Mass matrices
lodStiff = lod_periodic.assembleMsStiffnessMatrix(world, patchT, KmsijT, periodic=True)


# Fix the matrices for B.C.

# Compute for eigen values
def compute_eigen_vals(lodStiff, Mass)
    evals, evecs = eigh(lodStiff, Mass)
    return evals