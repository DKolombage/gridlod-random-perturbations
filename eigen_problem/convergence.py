from with_FEM_MassMatrix import *
from Reference_Solvers import *
import math
from numpy import *
from numpy.linalg import norm
import matplotlib.pyplot as plt
from mpltools import annotation
from offline_online_alg import *


#def solve_EVP(*solver, Neigen, NCoarse, NFine, Nepsilon, k, NSamples, pList,alpha,beta, model, reference_solver="FEM", save_files = True, plot=True, root=None):


def convergence(Neigen, NCoarse, NFine, Nepsilon, k, NSamples, pList,alpha,beta, model, solver , reference_solver="FEM", save_files = True, plot=True, root=None):

    Niter = 4
    NC_list = []
    rmserr_p_λ1 = []
    rmserr_p_λ2=[]

    for j in range(Niter):
        NCoarse *= 2
        if reference_solver == "FEM" and solver == "KOOLOD":
            K_λ1, K_λ2 = KOOLOD_MFEM_EigenSolver(NCoarse, NFine, Nepsilon, k, alpha, beta, NSamples, pList, Neigen, model, save_file=False) #KLOD_MFEM_EigenSolver(NCoarse, NFine, Nepsilon, k, alpha, beta, NSamples, pList, Neigen, save_file=False)
            M_λ1, M_λ2 =  FEM_EigenSolver(Neigen, NSamples, pList,alpha,beta, NCoarse, NFine, Nepsilon, save_file=False)
            absErrorList_λ1 = abs(K_λ1-M_λ1)  # p in rows and Nsamples in columns
            absErrorList_λ2 = abs(K_λ2-M_λ2)

        elif reference_solver == "FEM" and solver == "LOD":
            K_λ1, K_λ2 = KLOD_MFEM_EigenSolver(NCoarse, NFine, Nepsilon, k, alpha, beta, NSamples, pList, Neigen, model, save_file=False)
            M_λ1, M_λ2 =  FEM_EigenSolver(Neigen, NSamples, pList,alpha,beta, NCoarse, NFine, Nepsilon, save_file=False)
            absErrorList_λ1 = abs(K_λ1-M_λ1)  # p in rows and Nsamples in columns
            absErrorList_λ2 = abs(K_λ2-M_λ2)

        elif reference_solver == "LOD" and solver == "KOOLOD":
            K_λ1, K_λ2 = KOOLOD_MFEM_EigenSolver(NCoarse, NFine, Nepsilon, k, alpha, beta, NSamples, pList, Neigen, model, save_file=False)
            M_λ1, M_λ2 =  KLOD_MFEM_EigenSolver(NCoarse, NFine, Nepsilon, k, alpha, beta, NSamples, pList, Neigen, save_file=False)
            absErrorList_λ1 = abs(K_λ1-M_λ1)  # p in rows and Nsamples in columns
            absErrorList_λ2 = abs(K_λ2-M_λ2)

        elif reference_solver == "exact":
            Exact_λ = Exact_EigenSolver(Neigen)
            absErrorList_λ1 = abs(K_λ1-Exact_λ)  # p in rows and Nsamples in columns
            absErrorList_λ2 = abs(K_λ2-Exact_λ)
        else:
            print("Unrecognized reference solver!")

        rmserr_λ1 = np.sqrt(1. / NSamples * np.sum(absErrorList_λ1** 2, axis = 1))
        rmserr_λ2 = np.sqrt(1. / NSamples * np.sum(absErrorList_λ2** 2, axis = 1))
        rmserr_p_λ1.append(rmserr_λ1)
        rmserr_p_λ2.append(rmserr_λ2)
        #print("rmsp1", rmserr_p_λ1)
        #print("rmsp2", rmserr_p_λ2)
        NC_list.append(np.copy(NCoarse[0]))
        #print(NC_list)

        if save_files:
            if not root == None:
                sio.savemat(root + '_pList_NCList'+'.mat', {'pList': pList, 'NC_list': NC_list})
                sio.savemat(root + '_meanErr_H' + str(NCoarse[0]) + '.mat', {'absErr_1': absErrorList_λ1, 'absErr_2': absErrorList_λ2, 'pList': pList})
            else: 
                sio.savemat('_pList_NCList'+'.mat', {'pList': pList, 'NC_list': NC_list})
                sio.savemat('_meanErr_H' + str(NCoarse[0]) + '.mat', {'absErr_1': absErrorList_λ1, 'absErr_2': absErrorList_λ2, 'pList': pList})
    #print("check2", NC_list)
    err1 = np.array(rmserr_p_λ1)
    err2 = np.array(rmserr_p_λ2)
    #print("err1", err1)
    Nlines =3
    if save_files: 
        if not root == None:
            sio.savemat(root + '_RMSErr_H'  + '.mat', {'rmserr_lamb1': err1, 'rmserr_lamb2': err2, 'pList': pList, 'NC_list': NC_list})
        else:
            sio.savemat('_RMSErr_H'  + '.mat', {'rmserr_lamb1': err1, 'rmserr_lamb2': err2, 'pList': pList, 'NC_list': NC_list})
    if plot:
        ax1=plt.figure().add_subplot()
        ax2=plt.figure().add_subplot()
        #ax3 =plt.figure().add_subplot()
        for i in range(len(pList)):
            labelplain = 'p={' + str(pList[i]) + '}'
            ax1.loglog(NC_list, err1[:,i], label=r'${}$'.format(labelplain), marker='>')
            ax2.loglog(NC_list, err2[:,i], label=r'${}$'.format(labelplain), marker='<')
        ax1.legend()
        ax2.legend()
        #ax3.legend()
        ax1.set_xlabel('$H^{-1}$')
        ax1.set_ylabel('Root Mean squard error of $λ_1$')
        ax2.set_xlabel('$H^{-1}$')
        ax2.set_ylabel('Root Mean squard error of $λ_2$')
        plt.show()

        fig = plt.figure()
        ax4 = fig.add_subplot(1, 2, 1)
        ax5 = fig.add_subplot(1, 2, 2)

        i = -2
        print("Hplot", NC_list)
        for N in NC_list:
            if not root == None:
                err = sio.loadmat(root + '_meanErr_H' + str(N) + '.mat')
            else:
                err = sio.loadmat('_meanErr_H' + str(N) + '.mat')
            Error_λ1 = err['absErr_1']
            pList = err['pList'][0]
            Error_λ2 = err['absErr_2']
            NSamples = len(Error_λ2[0, :])
            rms_λ1 = []
            rms_λ2 = []
            for ii in range(len(pList)):
                rms_λ1.append(np.sqrt(1. / NSamples * np.sum(Error_λ1[ii, :] ** 2)))
                rms_λ2.append(np.sqrt(1. / NSamples * np.sum(Error_λ2[ii, :] ** 2)))
            labelplain = 'H=2^{' + str(i) + '}'
            ax4.plot(pList, rms_λ1, '-*', label=r'${}$'.format(labelplain))
            ax5.plot(pList, rms_λ2, '-*', label=r'${}$'.format(labelplain))
            i -= 1
        ax4.legend()
        ax5.legend()
        ax4.set_xlabel('p')
        ax4.set_ylabel('root means square error of $λ_1$')
        ax5.set_xlabel('p')
        ax5.set_ylabel('root means square error of $λ_2$')
        plt.show()
    return print("Root mean square absolute error of λ1:\n", err1), print("Root mean square absolute error of λ2: \n", err2) #err1, err2 , NC_list #

   