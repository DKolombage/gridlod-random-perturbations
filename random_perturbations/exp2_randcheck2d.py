import numpy as np
import scipy.io as sio
import time

from gridlod.world import World
from gridlod import util, fem, lod, interp
import algorithms, build_coefficient,lod_periodic

NFine = np.array([256, 256])
NpFine = np.prod(NFine+1)
Nepsilon = np.array([128,128])
NCoarse = np.array([32,32])
k=4
NSamples = 350
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
ffunc = lambda x: 8*np.pi**2*np.sin(2*np.pi*x[:,0])*np.cos(2*np.pi*x[:,1])
f = ffunc(xpFine).flatten()

aRefList, KmsijList,muTPrimeList, timeBasis, timeMatrixList = algorithms.computeCSI_offline(world, Nepsilon // NCoarse,
                                                                                            k,boundaryConditions,model)
aRef = np.copy(aRefList[-1])
KmsijRef = np.copy(KmsijList[-1])
muTPrimeRef = muTPrimeList[-1]

print('offline time for new approach {}'.format(timeBasis+np.sum(np.array(timeMatrixList))))
print('offline time for perturbed LOD {}'.format(timeMatrixList[-1]))

abserr_comb= np.zeros((len(pList), NSamples))
relerr_comb= np.zeros((len(pList), NSamples))
abserr_noup= np.zeros((len(pList), NSamples))
relerr_noup= np.zeros((len(pList), NSamples))
abserr_up = np.zeros((len(pList), NSamples))
relerr_up = np.zeros((len(pList), NSamples))

def computeKmsij(TInd, a, IPatch):
    patch = lod_periodic.PatchPeriodic(world, k, TInd)
    aPatch = lod_periodic.localizeCoefficient(patch,a, periodic=True)

    correctorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)
    csi = lod.computeBasisCoarseQuantities(patch, correctorsList, aPatch)
    return patch, correctorsList, csi.Kmsij, csi

# LOD for deterministic coeffcient - no updates
basis = fem.assembleProlongationMatrix(world.NWorldCoarse, world.NCoarseElement)
MFull = fem.assemblePatchMatrix(world.NWorldFine, world.MLocFine)
computePatch = lambda TInd: lod_periodic.PatchPeriodic(world, k, TInd)
patchT = list(map(computePatch, range(world.NtCoarse)))
KFullpert = lod_periodic.assembleMsStiffnessMatrix(world, patchT, KmsijRef, periodic=True)
bFull = basis.T * MFull * f
faverage = np.dot(MFull * np.ones(NpFine), f)
uFullpert, _ = lod_periodic.solvePeriodic(world, KFullpert, bFull, faverage, boundaryConditions)
uLodCoarsepert = basis * uFullpert

ii = 0
for p in pList:
    if p == 0.1:
        mean_time_true = 0.
        mean_time_perturbed = 0.
        mean_time_combined = 0.

    for N in range(NSamples):
        aPert = build_coefficient.build_randomcheckerboard(Nepsilon,NFine,alpha,beta,p)

        #true LOD
        middle = world.NWorldCoarse[1] // 2 * world.NWorldCoarse[0] + world.NWorldCoarse[0] // 2
        patchRef = lod_periodic.PatchPeriodic(world, k, middle)
        IPatch = lambda: interp.L2ProjectionPatchMatrix(patchRef)
        computeKmsijT = lambda TInd: computeKmsij(TInd, aPert, IPatch)
        if p == 0.1:
            tic = time.perf_counter()
            patchT, _, KmsijTtrue, _ = zip(*map(computeKmsijT, range(world.NtCoarse)))
            KFulltrue = lod_periodic.assembleMsStiffnessMatrix(world, patchT, KmsijTtrue, periodic=True)
            toc = time.perf_counter()
            mean_time_true += (toc-tic)
        else:
            patchT, _, KmsijTtrue, _ = zip(*map(computeKmsijT, range(world.NtCoarse)))
            KFulltrue = lod_periodic.assembleMsStiffnessMatrix(world, patchT, KmsijTtrue, periodic=True)

        bFull = basis.T * MFull * f
        uFulltrue, _ = lod_periodic.solvePeriodic(world, KFulltrue, bFull, faverage, boundaryConditions)
        uLodCoarsetrue = basis * uFulltrue

        #combined LOD
        if p == 0.1:
            tic = time.perf_counter()
            KFullcomb, _ = algorithms.compute_combined_MsStiffness(world,Nepsilon,aPert,aRefList,KmsijList,muTPrimeList,
                                                                   k,model,compute_indicator=False)
            toc = time.perf_counter()
            mean_time_combined += (toc-tic)
        else:
            KFullcomb, _ = algorithms.compute_combined_MsStiffness(world,Nepsilon,aPert,aRefList,KmsijList,muTPrimeList,
                                                                   k,model,compute_indicator=False)
        bFull = basis.T * MFull * f
        uFullcomb, _ = lod_periodic.solvePeriodic(world, KFullcomb, bFull, faverage, boundaryConditions)
        uLodCoarsecomb = basis * uFullcomb

        L2norm = np.sqrt(np.dot(uLodCoarsetrue, MFull * uLodCoarsetrue))
        abs_error_combined = np.sqrt(np.dot(uLodCoarsetrue - uLodCoarsecomb, MFull * (uLodCoarsetrue - uLodCoarsecomb)))
        abserr_comb[ii, N] = abs_error_combined
        relerr_comb[ii, N] = abs_error_combined / L2norm

        # standard LOD no updates
        abs_error_pert = np.sqrt(np.dot(uLodCoarsetrue - uLodCoarsepert, MFull * (uLodCoarsetrue - uLodCoarsepert)))
        abserr_noup[ii, N] = abs_error_pert
        relerr_noup[ii, N] = abs_error_pert / L2norm

        # LOD with updates
        if p == 0.1:
            tic = time.perf_counter()
            KFullpertup, _ = algorithms.compute_perturbed_MsStiffness(world, aPert, aRef, KmsijRef, muTPrimeRef, k,
                                                                      percentage_comp)
            toc = time.perf_counter()
            mean_time_perturbed += (toc - tic)
            bFull = basis.T * MFull * f
            uFullpertup, _ = lod_periodic.solvePeriodic(world, KFullpertup, bFull, faverage, boundaryConditions)
            uLodCoarsepertup = basis * uFullpertup
            error_pertup = np.sqrt(
                np.dot(uLodCoarsetrue - uLodCoarsepertup, MFull * (uLodCoarsetrue - uLodCoarsepertup)))
            abserr_up[ii, N] = error_pertup
            relerr_up[ii, N] = error_pertup / L2norm

    rmserrNew = np.sqrt(1. / NSamples * np.sum(relerr_comb[ii, :] ** 2))
    rmserrNoup = np.sqrt(1. / NSamples * np.sum(relerr_noup[ii, :] ** 2))
    rmserrUp = np.sqrt(1. / NSamples * np.sum(relerr_up[ii, :] ** 2))
    print("root mean square relative L2-error for new LOD over {} samples for p={} is: {}".format(NSamples,p,rmserrNew))
    print("root mean square relative L2-error for perturbed LOD without updates over {} samples for p={} is: {}".
          format(NSamples, p, rmserrNoup))
    if p == 0.1:
        print("root mean square relative L2-error for perturbed LOD with {} updates over {} samples for p={} is: {}".
              format(percentage_comp, NSamples, p, rmserrUp))

    ii += 1

    if p == 0.1:
        mean_time_true /= NSamples
        mean_time_perturbed /= NSamples
        mean_time_combined /= NSamples

        print("mean assembly time for standard LOD over {} samples is: {}".format(NSamples, mean_time_true))
        print("mean assembly time for perturbed LOD with {} updates over {} samples is: {}".
              format(NSamples, percentage_comp,mean_time_perturbed))
        print("mean assembly time for new LOD over {} samples is: {}".format(NSamples, mean_time_combined))

sio.savemat('_meanErr2drandcheck.mat', {'abserrNew': abserr_comb, 'relerrNew': relerr_comb,
                                        'absErrNoup': abserr_noup, 'relerrNoup': abserr_noup,
                                        'absErrUp': abserr_up, 'relerrUp': relerr_up, 'pList': pList})