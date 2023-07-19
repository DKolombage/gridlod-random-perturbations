from with_FEM_MassMatrix import *
from Reference_Solvers import *
import math
from numpy import *
from numpy.linalg import norm
import matplotlib.pyplot as plt
from mpltools import annotation

#Compute exact or reference eigen values
def compute_reference_eig(Neigen=Neigen, exact=False, ref_solver=None): #NCoarse=NCoarse, NFine=NFine, NSamples=None, pList=pList,
    if exact:
        eigenvalues = []
        for i in range(Neigen+1):
            if (i%2)==0:
                eig = (i**2)*math.pi**2
            else: 
                eig = ((i-1)**2)*math.pi**2
            eigenvalues.append(eig)
        return eigenvalues[2::] #return  print('Exact Eigenvalues:\n', eigenvalues[1::])                             
    else: # add for reference solver when exact is non-callable in higher dimensions
        eigenvalues = FEM_EigenSolver(Neigen, np.array([64]), np.array([256]), np.array([128]))
        return eigenvalues
    
C= compute_reference_eig(Neigen=3, exact=False, ref_solver=None)
print('c',C)

def convergence(Neigen, NCoarse, return_errors=False):
    errors = []
    N = 4
    Nlines = 3
    approx = []
    NC_list = []
    #NCoarse = np.array([8])
    #print('NC_list n\:', NC_list)
    #L=np.array([8, 16
    # ,32, 64,128])
    for i in range(N):
        NCoarse *= 2
        ref_eigenvalues = compute_reference_eig(Neigen)
        print('Reference values:\n', ref_eigenvalues)
        approx_eigenvalues = KLOD_MFEM_EigenSolver(NCoarse)
        # app = approx_eigenvalues[1:]
        print('approx:\n', approx_eigenvalues)
        approx.append(approx_eigenvalues)
        error = abs(ref_eigenvalues- approx_eigenvalues)
        errors.append(error)
        NC_list.append(np.copy(NCoarse[0]))
    print('NC_list n\:', NC_list)
    print('Approx list:\n', approx)
    print('Error list \n',errors)
    err = np.array(errors);
    print(err);
    ax=plt.figure().add_subplot()
    #ax = plt.figure().add_subplot()
    ax.loglog(NC_list, err[:,0], label='$λ_1$', marker='>')
    ax.loglog(NC_list, err[:,1], label='$λ_2$', marker='<')
    for j in range(1, Nlines+1):
        ax.loglog(NC_list, [err[0]*0.5**(i*j) for i in range(N)], lw = 0.5, color="grey")
    #axes = plt.gca()
    NC_list_mid = np.convolve(NC_list, [0.5, 0.5], mode='valid')
    err_mid = np.interp(NC_list_mid, NC_list, err[:,0])
    grad1 = np.diff(err[:,0])/np.diff(NC_list)
    for xm, ym, g in zip(NC_list_mid, err_mid, grad1):
        annotation.slope_marker((xm,ym), g)
    ax.set_xlabel("Number of Coarse steps")
    ax.set_ylabel("abs. error")
    ax.set_title("Convergence")
    ax.legend(loc='best')
    plt.show()
    if return_errors == False:
        return  
    else:
        return errors
#A=convergence(3, np.array([8]), return_errors = True)
#print(A)

NCoarse=np.array([8])
def EOC(NCoarse):
    Errors_List = np.array(convergence(3, np.array([8]), return_errors = True))
    NMeshRefinements = len(Errors_List[:,0])
    Nlist = np.array([2*np.array([8]), 4*np.array([8]), 8*np.array([8]), 16*np.array([8])])
    print(Nlist)
    #for i in range(NMeshRefinements):
    #    NCoarse = NCoarse*2
    #    Nlist.append(NCoarse[0])
    Alpha_list = []
    for i in range(NMeshRefinements-1):
        EOC_values = (np.log10((Errors_List[i]/Errors_List[i+1])))/(np.log10(Nlist[i]/Nlist[i+1]))
        Alpha_list.append(EOC_values)
    return print("EoC: \n",Alpha_list), Nlist
E = EOC(np.array([8]))
print(E)
#C=compute_reference_eig(Neigen=10, exact=True)
#print('exact', C)

