import numpy as np
import scipy.io as sio

from gridlod.world import World
from gridlod import util, fem, lod, interp
import algorithms, build_coefficient, lod_periodic

def computeKmsij(TInd, a, IPatch):
    patch = lod_periodic.PatchPeriodic(world, k, TInd)
    aPatch = lod_periodic.localizeCoefficient(patch,a, periodic=True)

    correctorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)
    csi = lod.computeBasisCoarseQuantities(patch, correctorsList, aPatch)
    return patch, correctorsList, csi.Kmsij, csi

#2d
NFine = np.array([40, 40])
NpFine = np.prod(NFine+1)
Nepsilon = np.array([20,20])
NCoarse = np.array([5,5])
k=2
NSamples = 500
dim = np.size(NFine)

xpFine = util.pCoordinates(NFine)
ffunc = lambda x: 8 * np.pi ** 2 * np.sin(2 * np.pi * x[:,0])*np.cos(2*np.pi*x[:,1])
f = ffunc(xpFine).flatten()

boundaryConditions = None
alpha = 0.1
beta = 1.
pList = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11]
np.random.seed(123)
model ={'name': 'check', 'alpha': alpha, 'beta': beta}

NCoarseElement = NFine // NCoarse
world = World(NCoarse, NCoarseElement, boundaryConditions)

aRefList, KmsijList, muTPrimeList, _, _, correctorsList\
    = algorithms.computeCSI_offline(world, Nepsilon // NCoarse, k, boundaryConditions,model, True)

middle = world.NWorldCoarse[1] // 2 * world.NWorldCoarse[0] + world.NWorldCoarse[0] // 2
patchRef = lod_periodic.PatchPeriodic(world, k, middle)

for p in pList:
    ETList = []
    ETListmiddle = []
    absErrorList = []
    relErrorList = []

    for N in range(NSamples):

        aPert = build_coefficient.build_randomcheckerboard(Nepsilon, NFine, alpha, beta, p)

        MFull = fem.assemblePatchMatrix(world.NWorldFine, world.MLocFine)
        basis = fem.assembleProlongationMatrix(world.NWorldCoarse, world.NCoarseElement)
        bFull = basis.T * MFull * f
        faverage = np.dot(MFull * np.ones(NpFine), f)

        IPatch = lambda: interp.L2ProjectionPatchMatrix(patchRef)
        computeKmsijT = lambda TInd: computeKmsij(TInd, aPert, IPatch)
        patchT, _, KmsijTtrue, _ = zip(*map(computeKmsijT, range(world.NtCoarse)))
        KFulltrue = lod_periodic.assembleMsStiffnessMatrix(world, patchT, KmsijTtrue, periodic=True)

        bFull = basis.T * MFull * f
        uFulltrue, _ = lod_periodic.solvePeriodic(world, KFulltrue, bFull, faverage, boundaryConditions)
        uLodCoarsetrue = basis * uFulltrue

        # combined LOD
        KFullcomb, indic = algorithms.compute_combined_MsStiffness(world,Nepsilon,aPert,aRefList,KmsijList,muTPrimeList,
                                                                   k,model,True,correctorsList)
        ETs = [indic[ii][0] for ii in range(len(indic))]
        ETs = np.array(ETs)
        ETList.append(ETs)
        ETListmiddle.append(ETs[middle])

        bFull = basis.T * MFull * f
        uFullcomb, _ = lod_periodic.solvePeriodic(world, KFullcomb, bFull, faverage, boundaryConditions)
        uLodCoarsecomb = basis * uFullcomb

        L2norm = np.sqrt(np.dot(uLodCoarsetrue, MFull * uLodCoarsetrue))
        error_combined = np.sqrt(
            np.dot(uLodCoarsetrue - uLodCoarsecomb, MFull * (uLodCoarsetrue - uLodCoarsecomb)))
        absErrorList.append(error_combined)
        relErrorList.append(error_combined/L2norm)

    rmserr = np.sqrt(1. / NSamples * np.sum(np.array(relErrorList) ** 2))
    rmsindic = np.sqrt(1. / NSamples * np.sum(np.array(ETListmiddle) ** 2))
    print("root mean square relative L2-error for p = {} is: {}".format(p, rmserr))
    print("root mean square error indicator element for p = {} is: {}".format(p, rmsindic))

    sio.savemat('data/_ErrIndic2drandcheck_p'+str(p)+'.mat', {'ETListloc': ETList, 'ETListmiddle': ETListmiddle,
                                                              'absError': absErrorList, 'relError': relErrorList})
