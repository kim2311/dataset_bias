from sklearn import decomposition
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

fmnist = np.load('features/MNIST.npy')
fmnistm = np.load('features/MNIST_M.npy')
fsvhn = np.load('features/SVHN.npy')
fusps = np.load('features/USPS.npy')
feats = np.concatenate([fmnist, fmnistm, fsvhn, fusps])

pca = decomposition.PCA(n_components=2)
pca.fit(feats)
coords = pca.transform(feats)

LABELS = ['MNIST', 'MNIST_M', 'SVHN', 'USPS']
MARKER = ['.', 'v', '*', 'x']
COLOR = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

for i in range(4):
    plt.scatter(coords[i * 100:i * 100 + 100, 0], coords[i * 100:i * 100 + 100, 1], color=COLOR[i], marker=MARKER[i],
                label=LABELS[i])
plt.legend(fontsize='x-large')
plt.savefig('pca1.jpg')
plt.close()

'''
mn_ms = []
mn_svs = []
mn_us = []
mm_svs = []
mm_us = []
sv_us = []

with open('pca_dist.txt', 'a') as myfile:
    for i in range(100):
        myfile.write('\n{:<10}{:<10}{:<10}{:<10}{:<10}\n'.format('', 'MNIST', 'MNIST-M', 'SVHN', 'USPS'))
        pca = decomposition.PCA(n_components=2)
        pca.fit(feats)
        coords = pca.transform(feats)

        mnist = np.average(coords[0:100], axis=0).reshape(1, -1)
        mnistm = np.average(coords[100:200], axis=0).reshape(1, -1)
        svhn = np.average(coords[200:300], axis=0).reshape(1, -1)
        usps = np.average(coords[300:400], axis=0).reshape(1, -1)

        mn_m = cdist(mnist, mnistm)[0][0]
        mn_sv = cdist(mnist, svhn)[0][0]
        mn_u = cdist(mnist, usps)[0][0]
        mm_sv = cdist(mnistm, svhn)[0][0]
        mm_u = cdist(mnistm, usps)[0][0]
        sv_u = cdist(svhn, usps)[0][0]

        mn_ms.append(mn_m)
        mn_svs.append(mn_sv)
        mn_us.append(mn_u)
        mm_svs.append(mm_sv)
        mm_us.append(mm_u)
        sv_us.append(sv_u)

        myfile.write('{:<10}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}\n'
                     .format('MNIST', 0, mn_m, mn_sv, mn_u))
        myfile.write('{:<10}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}\n'
                     .format('MNIST-M', mn_m, 0, mm_sv, mm_u))
        myfile.write('{:<10}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}\n'
                     .format('SVHN', mn_sv, mm_sv, 0, sv_u))
        myfile.write('{:<10}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}\n'
                     .format('USPS', mn_u, mm_u, sv_u, 0))

mn_m = np.average(np.asarray(mn_ms))
mn_sv = np.average(np.asarray(mn_svs))
mn_u = np.average(np.asarray(mn_us))
mm_sv = np.average(np.asarray(mm_svs))
mm_u = np.average(np.asarray(mm_us))
sv_u = np.average(np.asarray(sv_us))

with open('pca_avg.txt', 'w') as myfile:
    myfile.write('\n')
    myfile.write('Averaged:\n')
    myfile.write('{:<10}{:<10}{:<10}{:<10}{:<10}\n'.format('', 'MNIST', 'MNIST-M', 'SVHN', 'USPS'))
    myfile.write('{:<10}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}\n'
                 .format('MNIST', 0, mn_m, mn_sv, mn_u))
    myfile.write('{:<10}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}\n'
                 .format('MNIST-M', mn_m, 0, mm_sv, mm_u))
    myfile.write('{:<10}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}\n'
                 .format('SVHN', mn_sv, mm_sv, 0, sv_u))
    myfile.write('{:<10}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}\n'
                 .format('USPS', mn_u, mm_u, sv_u, 0))
'''
