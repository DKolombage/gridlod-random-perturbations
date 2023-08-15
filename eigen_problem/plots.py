from with_FEM_MassMatrix import *
from Reference_Solvers import *
import math
from numpy.linalg import norm
import matplotlib.pyplot as plt
from mpltools import annotation
import numpy as np
import matplotlib
from gridlod import util
from gridlod.world import World
sys.path.insert(0, '/home/kolombag/Documents/gridlod-random-perturbations/random_perturbations')
import build_coefficient, lod_periodic

np.random.seed(123)

def plots_cvg(root, H_Convergence=True, p_Convergence=True):
    pNC = sio.loadmat(root + '_pList_NCList' + '.mat')
    pList = pNC['pList'][0]
    NC_list = pNC['NC_list'][0]
    print("hplot", NC_list)
    if H_Convergence:
        ax1=plt.figure().add_subplot()
        ax2=plt.figure().add_subplot()
        data_array = sio.loadmat(root + '_RMSErr_H' + '.mat')
        err_Lam1 = data_array['rmserr_lamb1']
        err_Lam2 = data_array['rmserr_lamb2']
        for i in range(len(pList)):
            labelplain = 'p={' + str(pList[i]) + '}'
            ax1.loglog(NC_list, err_Lam1[:, i], label=r'${}$'.format(labelplain), marker='>')
            ax2.loglog(NC_list, err_Lam2[:,i], label=r'${}$'.format(labelplain), marker='<')
        ax1.legend()
        ax2.legend()
        #ax3.legend()
        ax1.set_xlabel('$H^{-1}$')
        ax1.set_ylabel('Root Mean squard error of $λ_1$')
        ax2.set_xlabel('$H^{-1}$')
        ax2.set_ylabel('Root Mean squard error of $λ_2$')
        plt.show()

    if p_Convergence:
        fig = plt.figure()
        ax4 = fig.add_subplot(1, 2, 1)
        ax5 = fig.add_subplot(1, 2, 2)

        i = -3
        for N in NC_list:
            err = sio.loadmat(root + '_meanErr_H' + str(N) + '.mat')
            Error_λ1 = err['absErr_1']
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
        

# Plot coefficients
# Same code from plot_coeff as a function

def plots_coeffs(Nepsilon, NFine, alpha, beta, pList, type):
    if type == "rand_checkerboard":
        for p in pList:
            aPert = build_coefficient.build_randomcheckerboard(Nepsilon,NFine,alpha,beta,p)
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 1, 1)

            apertGrid = aPert.reshape(NFine, order='C')
            im1 = ax1.imshow(apertGrid, origin='lower', extent=(0, 1, 0, 1), cmap='Greys')
            fig.colorbar(im1, ax=ax1)
            plt.show()
    elif type == "random_erasure":
        for p in pList:
            incl_bl = np.array([0.25, 0.25])
            incl_tr = np.array([0.75, 0.75])
            aPert = build_coefficient.build_inclusions_defect_2d(NFine,Nepsilon,alpha,beta,incl_bl,incl_tr,p)
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 1, 1)

            apertGrid = aPert.reshape(NFine, order='C')
            im1 = ax1.imshow(apertGrid, origin='lower', extent=(0, 1, 0, 1), cmap='Greys')
            fig.colorbar(im1, ax=ax1)
            plt.show()
    else:
        for p in pList:
            incl_bl = np.array([0.25, 0.25])
            incl_tr = np.array([0.75, 0.75])
            Lshape_bl = np.array([0.5, 0.5])
            Lshape_tr = np.array([0.75, 0.75])
            shift_bl = np.array([0.75, 0.75])
            shift_tr=np.array([1., 1.])
            model1={'name': 'inclfill'}
            model2={'name':'inclshift', 'def_bl': shift_bl, 'def_tr': shift_tr}
            model3={'name':'inclLshape', 'def_bl': Lshape_bl, 'def_tr': Lshape_tr}

            aPertList = []
            aPertList.append(build_coefficient.build_inclusions_defect_2d(NFine,Nepsilon,alpha,beta,incl_bl,incl_tr,p))
            aPertList.append(build_coefficient.build_inclusions_defect_2d(NFine,Nepsilon,alpha,beta,incl_bl,incl_tr,p, 0.5))
            aPertList.append(build_coefficient.build_inclusions_defect_2d(NFine,Nepsilon,alpha,beta,incl_bl,incl_tr,p, 5.))
            aPertList.append(build_coefficient.build_inclusions_change_2d(NFine,Nepsilon,alpha,beta,incl_bl,incl_tr,p, model1))
            aPertList.append(build_coefficient.build_inclusions_change_2d(NFine,Nepsilon,alpha,beta,incl_bl,incl_tr,p, model2))
            aPertList.append(build_coefficient.build_inclusions_change_2d(NFine,Nepsilon,alpha,beta,incl_bl,incl_tr,p, model3))
            fig = plt.figure()

            for ii in range(6):
                ax = fig.add_subplot(2, 3, ii+1)
                bounds = np.array([0.2, 0.8, 1.5, 5.5, 10.5])
                mycmap = plt.cm.get_cmap('Greys')
                norm = matplotlib.colors.BoundaryNorm(bounds, mycmap.N)
                apertGrid = aPertList[ii].reshape(NFine, order='C')
                im = ax.imshow(apertGrid, origin='lower', extent=(0, 1, 0, 1), cmap=mycmap, norm=norm)

            plt.show()
