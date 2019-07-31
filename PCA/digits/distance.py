import numpy as np
from scipy.spatial.distance import cdist

mnist = np.load('features/MNIST.npy')
mnistm = np.load('features/MNIST_M.npy')
svhn = np.load('features/SVHN.npy')
usps = np.load('features/USPS.npy')

mnist = np.average(mnist, axis=1).reshape((1, -1))
mnistm = np.average(mnistm, axis=1).reshape((1, -1))
svhn = np.average(svhn, axis=1).reshape((1, -1))
usps = np.average(usps, axis=1).reshape((1, -1))

print(cdist(mnist, mnistm))
print(cdist(mnist, svhn))
print(cdist(mnist, usps))
print(cdist(mnistm, svhn))
print(cdist(mnistm, usps))
print(cdist(svhn, usps))


