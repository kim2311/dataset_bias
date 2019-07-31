import numpy as np
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE

fimnet = np.load('feature_vectors/ImageNet.npy')
fvoc = np.load('feature_vectors/VOC.npy')
fcoco = np.load('feature_vectors/COCO.npy')
fsun = np.load('feature_vectors/SUN.npy')
fkitti = np.load('feature_vectors/KITTI.npy')
feats = np.concatenate([fimnet, fvoc, fcoco, fsun, fkitti])
print(feats.shape)

im_vocs = []
im_cocos = []
im_suns = []
im_kittis = []
voc_cocos = []
voc_suns = []
voc_kittis = []
coco_suns = []
coco_kittis = []
sun_kittis = []


with open('dist.txt', 'a') as myfile:
    myfile.write('\n')
    for lr in range(10, 301, 10):
        for perp in range(5, 51, 3):
            myfile.write('Learning Rate {}, Perplexity {}\n'.format(lr, perp))
            myfile.write('{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}\n'.format('','ImageNet', 'VOC', 'COCO', 'SUN', 'KITTI'))

            tsne = TSNE(learning_rate=lr, random_state=0, perplexity=perp, n_iter=1500)
            coords = tsne.fit_transform(feats)

            imnet = np.average(coords[0:100], axis=0).reshape(1, -1)
            voc = np.average(coords[100:200], axis=0).reshape(1, -1)
            coco = np.average(coords[200:300], axis=0).reshape(1, -1)
            sun = np.average(coords[300:400], axis=0).reshape(1, -1)
            kitti = np.average(coords[400:500], axis=0).reshape(1, -1)

            im_voc = cdist(imnet, voc)[0][0]
            im_coco = cdist(imnet, coco)[0][0]
            im_sun = cdist(imnet, sun)[0][0]
            im_kitti = cdist(imnet, kitti)[0][0]
            voc_coco = cdist(voc, coco)[0][0]
            voc_sun = cdist(voc, sun)[0][0]
            voc_kitti = cdist(voc, kitti)[0][0]
            coco_sun = cdist(coco, sun)[0][0]
            coco_kitti = cdist(coco, kitti)[0][0]
            sun_kitti = cdist(sun, kitti)[0][0]

            im_vocs.append(im_voc)
            im_cocos.append(im_coco)
            im_suns.append(im_sun)
            im_kittis.append(im_kitti)
            voc_cocos.append(voc_coco)
            voc_suns.append(voc_sun)
            voc_kittis.append(voc_kitti)
            coco_suns.append(coco_sun)
            coco_kittis.append(coco_kitti)
            sun_kittis.append(sun_kitti)

            myfile.write('{:<10}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}\n'
                         .format('ImageNet', 0, im_voc, im_coco, im_sun, im_kitti))
            myfile.write('{:<10}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}\n'
                         .format('VOC', im_voc, 0, voc_coco, voc_sun, voc_kitti))
            myfile.write('{:<10}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}\n'
                         .format('COCO', im_coco, voc_coco, 0, coco_sun, coco_kitti))
            myfile.write('{:<10}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}\n'
                         .format('SUN', im_sun, voc_sun, coco_sun, 0, sun_kitti))
            myfile.write('{:<10}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}\n\n\n'
                         .format('KITTI', im_kitti, voc_kitti, coco_kitti, sun_kitti, 0))

im_cocos = np.asarray(im_cocos)
im_vocs = np.asarray(im_vocs)
im_suns = np.asarray(im_suns)
im_kittis = np.asarray(im_kittis)
voc_cocos = np.asarray(voc_cocos)
voc_suns = np.asarray(voc_suns)
voc_kittis = np.asarray(voc_kittis)
coco_suns = np.asarray(coco_suns)
coco_kittis = np.asarray(coco_kittis)
sun_kittis = np.asarray(sun_kittis)

with open('avg.txt', 'a') as myfile:
    myfile.write('\n')
    myfile.write('{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}\n'.format('', 'ImageNet', 'VOC', 'COCO', 'SUN', 'KITTI'))
    myfile.write('{:<10}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}\n'
                 .format('ImageNet', 0, np.average(im_vocs),
                         np.average(im_cocos), np.average(im_suns), np.average(im_kittis)))
    myfile.write('{:<10}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}\n'
                 .format('VOC', np.average(im_vocs), 0, np.average(voc_cocos),
                         np.average(voc_suns), np.average(voc_kittis)))
    myfile.write('{:<10}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}\n'
                 .format('COCO', np.average(im_cocos), np.average(voc_cocos), 0,
                         np.average(coco_suns), np.average(coco_kittis)))
    myfile.write('{:<10}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}\n'
                 .format('SUN', np.average(im_suns), np.average(voc_suns),
                         np.average(coco_suns), 0, np.average(sun_kittis)))
    myfile.write('{:<10}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}\n\n\n'
                 .format('KITTI', np.average(im_kittis), np.average(voc_kittis),
                         np.average(coco_kittis), np.average(sun_kittis), 0))

