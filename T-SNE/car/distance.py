import numpy as np
from scipy.spatial.distance import cdist, cosine

LABELS = ['ImageNet', 'VOC', 'COCO', 'SUN', 'KITTI']
'''
imnet = np.load('feature_vectors/ImageNet.npy')
voc = np.load('feature_vectors/VOC.npy')
coco = np.load('feature_vectors/COCO.npy')
sun = np.load('feature_vectors/SUN.npy')
kitti = np.load('feature_vectors/KITTI.npy')

imnet = np.average(imnet, axis=1).reshape((1, -1))
voc = np.average(voc, axis=1).reshape((1, -1))
coco = np.average(coco, axis=1).reshape((1, -1))
sun = np.average(sun, axis=1).reshape((1, -1))
kitti = np.average(kitti, axis=1).reshape((1, -1))
print(imnet.shape, voc.shape, coco.shape, sun.shape, kitti.shape)

print(cdist(sun, kitti))
'''

coords = np.load('imgs/coords.npy')
print(coords.shape)

imnet = coords[0:100]
voc = coords[100:200]
coco = coords[200:300]
sun = coords[300:400]
kitti = coords[400:500]


imnet = np.average(imnet, axis=0).reshape(1, -1)
voc = np.average(voc, axis=0).reshape(1, -1)
coco = np.average(coco, axis=0).reshape(1, -1)
sun = np.average(sun, axis=0).reshape(1, -1)
kitti = np.average(kitti, axis=0).reshape(1, -1)

print(cdist(kitti, sun)[0][0])

#dist = []

'''
for i in range(100):
    for j in range(100):
        dist.append(cosine(sun[i], kitti[j]))
'''

#print(sum(dist) / len(dist))
