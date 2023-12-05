from with_FEM_MassMatrix import *
from convergence import *
from Reference_Solvers import *
from Experimental_Order_Cvg import *
from plots import *
import numpy as np

root = 'Data/k_layer_convergence/'

def k_convergence_plots( dimension=1, k=3, root=root):
    if dimension == 1 and k==4: 
        err1 = sio.loadmat(root + '_RC'+ '_rmserr_lamb_1_' + 'D'+ str(dimension) + 'k'+ str(k) + '.mat')
        err2 = sio.loadmat(root + '_RC'+ '_rmserr_lamb_2_' + 'D'+ str(dimension) + 'k'+ str(k) + '.mat')
        l1_k1_error = err1['err_k1']
        l1_k2_error = err1['err_k2']
        l1_k3_error = err1['err_k3']
        l1_k4_error = err1['err_k4']

        l2_k1_error = err2['err_k1']
        l2_k2_error = err2['err_k2']
        l2_k3_error = err2['err_k3']
        l2_k4_error = err2['err_k4']

        ts=np.array([16,32,64,128])
        #ts=np.array([8,16,32,64])
        plot(ts, l1_k1_error, l1_k2_error, l1_k3_error, l1_k4_error, names =["$k=1$", "$k=2$", "$k=3$", "$k=4$"], ylabel="Root mean squard error of $\lambda_1$", title = "LOD (against FEM) $k-$ layer convergence ($1-$D case with nk=4, $p=0.01$ and n\epsilon = 256, nh=512)")
        plot(ts,  l2_k1_error, l2_k2_error, l2_k3_error, l2_k4_error, names =["$k=1$", "$k=2$", "$k=3$", "$k=4$"], ylabel="Root mean squard error of $\lambda_2$", title = "LOD (against FEM) $k-$ layer convergence ($1-$D case with nk=4, $p=0.01$ and n\epsilon = 256, nh=512)")

    elif dimension == 1 and k==3: 
        err1 = sio.loadmat(root +'_RC'+ '_rmserr_lamb_1_' + 'D'+ str(dimension) + 'k'+ str(k) + '.mat')
        err2 = sio.loadmat(root +'_RC'+ '_rmserr_lamb_2_' + 'D'+ str(dimension) + 'k'+ str(k) + '.mat')
        l1_k1_error = err1['err_k1']
        l1_k2_error = err1['err_k2']
        l1_k3_error = err1['err_k3']

        l2_k1_error = err2['err_k1']
        l2_k2_error = err2['err_k2']
        l2_k3_error = err2['err_k3']

        ts=np.array([8, 16,32,64])
        #ts=np.array([8,16,32,64])
        plot(ts, l1_k1_error, l1_k2_error, l1_k3_error, names =["$k=1$", "$k=2$", "$k=3$"], ylabel="Root mean squard error of $\lambda_1$", title = "LOD (against FEM) $k-$ layer convergence ($1-$D case with nk=3, $p=0.01$ and n\epsilon = 128, nh=256)")
        plot(ts,  l2_k1_error, l2_k2_error, l2_k3_error, names =["$k=1$", "$k=2$", "$k=3$"], ylabel="Root mean squard error of $\lambda_2$", title = "LOD (against FEM) $k-$ layer convergence ($1-$D case with nk=3, $p=0.01$ and $n\epsilon = 128$, nh=256)")

    elif dimension == 2 and k==3: 
        err1= sio.loadmat(root + '_RC' + '_rmserr_lamb_1_' + 'D'+ str(dimension) + 'k'+ str(k) + '.mat')
        err2 = sio.loadmat(root + '_RC' + '_rmserr_lamb_2_' + 'D'+ str(dimension) + 'k'+ str(k) + '.mat')
        l1_k1_error = err1['err_k1']
        l1_k2_error = err1['err_k2']
        l1_k3_error = err1['err_k3']

        l2_k1_error = err2['err_k1']
        l2_k2_error = err2['err_k2']
        l2_k3_error = err2['err_k3']

        ts=np.array([8,16,32, 64])
        #ts=np.array([8,16,32,64])
        plot(ts, l1_k1_error, l1_k2_error, l1_k3_error, names =["$k=1$", "$k=2$", "$k=3$"], ylabel="Root mean squard error of $\lambda_1$", title="LOD (against FEM) $k-$ layer convergence ($2-$D case with nk=3, $p=0.01$ and n\epsilon = 128, nh=256)")
        plot(ts,  l2_k1_error, l2_k2_error, l2_k3_error, names =["$k=1$", "$k=2$", "$k=3$"], ylabel="Root mean squard error of $\lambda_2$", title="LOD (against FEM) $k-$ layer convergence ($2-$D case with nk=3, $p=0.01$ and  n\epsilon = 128, nh=256)")

    else:
        print("Other combinations are not considered!")

root = 'Data/k_layer_convergence/'
def k_convergence(dimension, k, root):
    alpha = 0.1
    beta = 1.
    NSamples =1
    pList = [0.01]
    Neigen = 3
    if dimension == 1 and k==4: 
        model ={'name': 'check', 'alpha': alpha, 'beta': beta}
        NCoarse = np.array([8])
        Nepsilon = np.array([256])
        NFine = np.array([512])    
        np.random.seed(123)

        print("one dimensional with $nk=4$ and $nH=16$, $n\epsilon = 256$, $nh=512$")
        err_k1 = errors(Neigen, NCoarse, NFine, Nepsilon, 1, NSamples, pList,alpha,beta, model, solver  = "LOD", reference_solver="FEM", save_files = False, root=None)
        err_l1_k1 = err_k1[0]
        err_l2_k1 = err_k1[1]

        NCoarse = np.array([8])
        Nepsilon = np.array([256])
        NFine = np.array([512])    
        err_k2 = errors(Neigen, NCoarse, NFine, Nepsilon, 2, NSamples, pList,alpha,beta, model, solver = "LOD", reference_solver="FEM", save_files = False, root=None)
        err_l1_k2 = err_k2[0]
        err_l2_k2 = err_k2[1]

        NCoarse = np.array([8])
        Nepsilon = np.array([256])
        NFine = np.array([512])    
        err_k3 = errors(Neigen, NCoarse, NFine, Nepsilon, 3, NSamples, pList,alpha,beta, model, solver = "LOD", reference_solver="FEM", save_files = False, root=None)
        err_l1_k3 = err_k3[0]
        err_l2_k3 = err_k3[1]

        NCoarse = np.array([8])
        Nepsilon = np.array([256])
        NFine = np.array([512])     
        err_k4 = errors(Neigen, NCoarse, NFine, Nepsilon, 4, NSamples, pList,alpha,beta, model, solver = "LOD" , reference_solver="FEM", save_files = False, root=None)
        err_l1_k4 = err_k4[0]
        err_l2_k4 = err_k4[1]

        sio.savemat(root + '_RC'+'_rmserr_lamb_1_' + 'D'+ str(dimension) + 'k'+ str(k) + '.mat', {'err_k1': err_l1_k1, 'err_k2': err_l1_k2, 'err_k3': err_l1_k3, 'err_k4':err_l1_k4}) #RC= Random Checkerboard
        sio.savemat(root + '_RC'+ '_rmserr_lamb_2_' + 'D'+ str(dimension) + 'k'+ str(k) + '.mat', {'err_k1': err_l2_k1, 'err_k2': err_l2_k2, 'err_k3': err_l2_k3, 'err_k4':err_l2_k4})
    
    elif dimension == 1 and k==3: 
        model ={'name': 'check', 'alpha': alpha, 'beta': beta}
        NCoarse = np.array([4])
        Nepsilon = np.array([128])
        NFine = np.array([256])    
        np.random.seed(123)
        root = 'Data/k_layer_convergence/'

        print("one dimensional with $nk=3$ and $nH=8$, $n\epsilon = 128$, $nh=256$")
        err_k1 = errors(Neigen, NCoarse, NFine, Nepsilon, 1, NSamples, pList,alpha,beta, model, solver  = "LOD", reference_solver="FEM", save_files = False, root=None)
        err_l1_k1 = err_k1[0]
        err_l2_k1 = err_k1[1]

        NCoarse = np.array([4])
        Nepsilon = np.array([128])
        NFine = np.array([256])    
        err_k2 = errors(Neigen, NCoarse, NFine, Nepsilon, 2, NSamples, pList,alpha,beta, model, solver = "LOD", reference_solver="FEM", save_files = False, root=None)
        err_l1_k2 = err_k2[0]
        err_l2_k2 = err_k2[1]

        NCoarse = np.array([4])
        Nepsilon = np.array([128])
        NFine = np.array([256])    
        err_k3 = errors(Neigen, NCoarse, NFine, Nepsilon, 3, NSamples, pList,alpha,beta, model, solver = "LOD", reference_solver="FEM", save_files = False, root=None)
        err_l1_k3 = err_k3[0]
        err_l2_k3 = err_k3[1]

        sio.savemat(root +'_RC'+ '_rmserr_lamb_1_' + 'D'+ str(dimension) + 'k'+ str(k) + '.mat', {'err_k1': err_l1_k1, 'err_k2': err_l1_k2, 'err_k3': err_l1_k3})
        sio.savemat(root + '_RC'+'_rmserr_lamb_2_' + 'D'+ str(dimension) + 'k'+ str(k) + '.mat', {'err_k1': err_l2_k1, 'err_k2': err_l2_k2, 'err_k3': err_l2_k3})

    elif dimension == 2 and k==3: 
        model ={'name': 'check', 'alpha': alpha, 'beta': beta}
        NCoarse = np.array([8,8])
        Nepsilon = np.array([128, 128])
        NFine = np.array([256, 256])    
        Neigen = 3
        np.random.seed(123)
        root = 'Data/k_layer_convergence/'

        print("Two dimensional with nk=3 and nH=8, n\epsilon = 128, nh=256")
        err_k1 = errors(Neigen, NCoarse, NFine, Nepsilon, 1, NSamples, pList,alpha,beta, model, solver  = "LOD", reference_solver="FEM", save_files = False, root=None)
        err_l1_k1 = err_k1[0]
        err_l2_k1 = err_k1[1]

        NCoarse = np.array([8,8])
        Nepsilon = np.array([128, 128])
        NFine = np.array([256, 256])     
        err_k2 = errors(Neigen, NCoarse, NFine, Nepsilon, 2, NSamples, pList,alpha,beta, model, solver = "LOD", reference_solver="FEM", save_files = False, root=None)
        err_l1_k2 = err_k2[0]
        err_l2_k2 = err_k2[1]

        NCoarse = np.array([8,8])
        Nepsilon = np.array([128, 128])
        NFine = np.array([256, 256])   
        err_k3 = errors(Neigen, NCoarse, NFine, Nepsilon, 3, NSamples, pList,alpha,beta, model, solver = "LOD", reference_solver="FEM", save_files = False, root=None)
        err_l1_k3 = err_k3[0]
        err_l2_k3 = err_k3[1]

        sio.savemat(root + '_RC'+'_rmserr_lamb_1_' + 'D'+ str(dimension) + 'k'+ str(k) + '.mat', {'err_k1': err_l1_k1, 'err_k2': err_l1_k2, 'err_k3': err_l1_k3})
        sio.savemat(root + '_RC'+'_rmserr_lamb_2_' + 'D'+ str(dimension) + 'k'+ str(k) + '.mat', {'err_k1': err_l2_k1, 'err_k2': err_l2_k2, 'err_k3': err_l2_k3})

    else:
        print("Only allowed k=3,4 and dimension =1, 2")

    return

