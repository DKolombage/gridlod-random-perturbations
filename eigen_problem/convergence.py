from with_FEM_MassMatrix import *
import math
from numpy import *
from numpy.linalg import norm
import matplotlib.pyplot as plt

#Compute exact or reference eigen values
def compute_reference_eig(Neigen=Neigen, exact=True, NCoarse=NCoarse, NFine=NFine, NSamples=None, pList=None, ref_solver=None):
    if exact:
        eigenvalues = []
        for i in range(Neigen+1):
            if (i%2)==0:
                eig = (i**2)*math.pi**2
            else: 
                eig = ((i-1)**2)*math.pi**2
            eigenvalues.append(eig)
        return eigenvalues[1::] #return  print('Exact Eigenvalues:\n', eigenvalues[1::])                             
    else: # add for reference solver when exact is non-callable in higher dimensions
        eigenvalues = FEM_EigenSolver(NCoarse, NFine, NSamples, pList,Neigen)
        return eigenvalues
    
#compute_reference_eig(3, exact=True, ref_solver=None)


def convergence(Neigen, NCoarse):
    errors = []
    N = 5
    approx = []
    NC_list = []
    L=np.array([8,16,32,64])
    for i in range(1,N):
        NCoarse *= 2
        ref_eigenvalues = compute_reference_eig(Neigen)
        print('exact:\n', ref_eigenvalues)
        approx_eigenvalues = KLOD_MFEM_EigenSolver(NCoarse)
        print('approx:\n', approx_eigenvalues)
        approx.append(approx_eigenvalues)
        error = abs(ref_eigenvalues- approx_eigenvalues)
        errors.append(error)
        NC_list.append(NCoarse)
    print('Approx list:\n', approx)
    print('error list \n',errors)
    print('NC_list n\:', NC_list)
    ax=plt.figure().add_subplot()
    ax.loglog(L, errors, label=['$λ_0$', '$λ_1$', '$λ_2$', '$λ_3$', '$λ_4$'])
    ax.set_xlabel("NCoarse steps")
    ax.set_ylabel("abs. error")
    ax.set_title("Convergence")
    ax.legend(loc='best')
    plt.show()
    return  

convergence(5, np.array([8]))

