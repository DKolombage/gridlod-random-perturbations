import numpy as np
import time

from gridlod import interp, lod
import build_coefficient, lod_periodic, indicator



def computeCSI_offline(world, NepsilonElement, k, boundaryConditions, model, correctors=False):
    ''' PatchPeriodic - 
    '''
    dim = np.size(world.NWorldFine)  # Di: Th difference between NFine and NWorldFine ?
    if dim == 2:
        middle = world.NWorldCoarse[1] // 2 * world.NWorldCoarse[0] + world.NWorldCoarse[0] // 2
    elif dim == 1:
        middle = world.NWorldCoarse[0] //2
    patch = lod_periodic.PatchPeriodic(world, k, middle)  

    tic = time.perf_counter()
    assert(model['name'] in ['check', 'incl', 'inclvalue', 'inclfill', 'inclshift', 'inclLshape'])
    if model['name'] == 'check':
        aRefList = build_coefficient.build_checkerboardbasis(patch.NPatchCoarse, NepsilonElement,
                                                             world.NCoarseElement, model['alpha'], model['beta'])
    elif model['name'] == 'incl':
        aRefList = build_coefficient.build_inclusionbasis_2d(patch.NPatchCoarse,NepsilonElement, world.NCoarseElement,
                                                             model['bgval'], model['inclval'], model['left'], model['right'])
    elif model['name'] == 'inclvalue':
        aRefList = build_coefficient.build_inclusionbasis_2d(patch.NPatchCoarse, NepsilonElement, world.NCoarseElement,
                                                             model['bgval'], model['inclval'], model['left'],
                                                             model['right'], model['defval'])
    elif model['name'] in ['inclfill', 'inclshift', 'inclLshape']:
        aRefList = build_coefficient.build_inclusionbasis_change_2d(patch.NPatchCoarse, NepsilonElement, world.NCoarseElement,
                                                             model['bgval'], model['inclval'], model['left'],
                                                             model['right'], model)

    toc = time.perf_counter()
    time_basis = toc-tic

    def computeKmsij(TInd, aPatch, k, boundaryConditions):
        ''' Di: nodalPatchMatrix -
                csi.muTPrime -
        '''
        tic = time.perf_counter()
        patch = lod_periodic.PatchPeriodic(world, k, TInd)
        if dim == 1:
            IPatch = lambda: interp.nodalPatchMatrix(patch)
        else:
            IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, boundaryConditions)

        correctorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)
        csi = lod.computeBasisCoarseQuantities(patch, correctorsList, aPatch)
        toc = time.perf_counter()
        return patch, correctorsList, csi.Kmsij, csi.muTPrime, toc-tic

    computeSingleKms = lambda aRef: computeKmsij(middle, aRef, k, boundaryConditions)
    if correctors:
        _, correctorsList, KmsijList, muTPrimeList, timeMatrixList = zip(*map(computeSingleKms, aRefList))
        return aRefList, KmsijList, muTPrimeList, time_basis, timeMatrixList, correctorsList
    else:
        _, _, KmsijList, muTPrimeList, timeMatrixList = zip(*map(computeSingleKms, aRefList))
        return aRefList, KmsijList, muTPrimeList, time_basis, timeMatrixList