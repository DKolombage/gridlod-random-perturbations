import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

root= 'data/'

# 1d
NList1d = [8, 16, 32, 64]
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

i = -3
for N in NList1d:
    err = sio.loadmat(root + '_meanErr_Nc' + str(N) + '.mat')
    harmerr = err['HarmErr']
    pList = err['pList'][0]
    relL2err = err['relErr']
    NSamples = len(relL2err[0, :])
    rms_harm = []
    rms_relerr = []
    for ii in range(len(pList)):
        rms_harm.append(np.sqrt(1. / NSamples * np.sum(harmerr[ii, :] ** 2)))
        rms_relerr.append(np.sqrt(1. / NSamples * np.sum(relL2err[ii, :] ** 2)))
    labelplain = 'H=2^{' + str(i) + '}'
    ax1.plot(pList, rms_harm, '-*', label=r'${}$'.format(labelplain))
    ax2.plot(pList, rms_relerr, '-*', label=r'${}$'.format(labelplain))
    i -= 1
ax1.legend()
ax2.legend()
ax1.set_xlabel('p')
ax1.set_ylabel('root means square $L^\infty$-error of harmonic means')
ax2.set_xlabel('p')
ax2.set_ylabel('root means square relative $L^2$-error of solutions')
plt.show()

# 2d random checkerboard

err2d = sio.loadmat(root + '_meanErr2drandcheck.mat')
errNew = err2d['relerrNew']
pList2d = err2d['pList'][0]
relL2errpert = err2d['relerrNoup']
rmsNew = []
rmsNoup = []
NSamples = len(errNew[0, :])
for ii in range(len(pList2d)):
    rmsNew.append(np.sqrt(1. / NSamples * np.sum(errNew[ii, :] ** 2)))
    rmsNoup.append(np.sqrt(1. / NSamples * np.sum(relL2errpert[ii, :] ** 2)))
    if pList2d[ii] == 0.1:
        relerrUp = err2d['relerrUp'][ii]
        rmsUp = np.sqrt(1. / NSamples * np.sum(relerrUp ** 2))
        print('root mean square relative $L^2$-error of LOD with 15% updates is {}'.format(rmsUp))

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

ax1.plot(pList2d, rmsNew, '-*')
ax1.set_xlabel('p')
ax1.set_ylabel('root mean square relative $L^2$-error')
ax2.plot(pList2d, rmsNoup, '-*')
ax2.set_xlabel('p')
ax2.set_ylabel('root mean square relative $L^2$-error')
plt.show()

# random defect change in value
def_values = [1., 0.5, 5.]
colors = ['b', 'r', 'g']
pList2d = [0.01, 0.05, 0.1, 0.15]
for ii in range(len(def_values)):
    err2d_value = sio.loadmat(root + '_meanErr2d_defvalues' + str(def_values[ii]) + '.mat')
    errNew = err2d_value['relerrDefect']
    rmsNew = []
    NSamples = len(errNew[0, :])
    for jj in range(len(pList2d)):
        rmsNew.append(np.sqrt(1. / NSamples * np.sum(errNew[jj, :] ** 2)))
    plt.plot(pList2d, rmsNew, colors[ii] + '-o', label='value=' + str(def_values[ii]))
plt.legend()
plt.xlabel('p')
plt.ylabel('root mean square relative $L^2$-errors')
plt.show()

# random defect change in geometry
changes = ['inclshift', 'inclfill', 'inclLshape']
names = [changes[j][4:] for j in range(len(changes))]
colors = ['b', 'r', 'g']
for ii in range(len(changes)):
    err2d_change = sio.loadmat(root + '_meanErr2d_defchanges' + changes[ii] + '.mat')
    errNew = err2d_change['relerr']
    pList2d = err2d_change['pList'][0]
    rmsNew = []
    NSamples = len(errNew[0, :])
    for jj in range(len(pList2d)):
        rmsNew.append(np.sqrt(1. / NSamples * np.sum(errNew[jj, :] ** 2)))
    plt.plot(pList2d, rmsNew, colors[ii] + '-o', label='model ' + str(names[ii]))
plt.legend()
plt.xlabel('p')
plt.ylabel('root mean square relative $L^2$-errors')
plt.show()

# indicator
pList = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11]

indicMiddleMean = []
errabsMean = []
errrelMean = []

for p in pList:
    errIndic = sio.loadmat(root + '_ErrIndic2drandcheck_p' + str(p) + '.mat')
    errabsList = errIndic['absError'][0]
    errrelList = errIndic['relError'][0]
    NSamples = len(errabsList)
    IndicListmiddle = errIndic['ETListmiddle'][0]

    indicMiddleMean.append(np.sqrt(1. / NSamples * np.sum(IndicListmiddle ** 2)))
    errabsMean.append(np.sqrt(1. / NSamples * np.sum(errabsList ** 2)))
    errrelMean.append(np.sqrt(1. / NSamples * np.sum(errrelList ** 2)))

errrelMean = np.array(errrelMean)
plt.figure()
plt.plot(pList, indicMiddleMean, 'b-*', label='$E_T$')
plt.plot(pList, errabsMean, 'r-*', label='absolute $L^2$-error')
plt.plot(pList, errrelMean, 'g-*', label='relative $L^2$-error')
plt.plot(pList, 4 * errrelMean, 'g--*', label='relative $L^2$-error multplied with $4$')
plt.legend()
plt.xlabel('p')
plt.ylabel('root mean square over 500 samples')

plt.show()
