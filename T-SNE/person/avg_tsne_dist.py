import numpy as np
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE

LABELS = ['CAM2', 'INRIA', 'VOC', 'COCO', 'SUN', 'KITTI', 'Caltech']

fcam2 = np.load('feature_vectors/CAM2.npy')
finria = np.load('feature_vectors/INRIA.npy')
fvoc = np.load('feature_vectors/VOC.npy')
fcoco = np.load('feature_vectors/COCO.npy')
fsun = np.load('feature_vectors/SUN.npy')
fkitti = np.load('feature_vectors/KITTI.npy')
fcal = np.load('feature_vectors/Caltech.npy')
feats = np.concatenate([fcam2, finria, fvoc, fcoco, fsun, fkitti, fcal])
print(feats.shape)


cam_inrias = []
cam_vocs = []
cam_cocos = []
cam_suns = []
cam_kittis = []
cam_cals = []
inria_vocs = []
inria_cocos = []
inria_suns = []
inria_kittis = []
inria_cals = []
voc_cocos = []
voc_suns = []
voc_kittis = []
voc_cals = []
coco_suns = []
coco_kittis = []
coco_cals = []
sun_kittis = []
sun_cals = []
kitti_cals = []



with open('dist.txt', 'a') as myfile:
    myfile.write('\n')
    for lr in range(10, 301, 10):
        for perp in range(5, 51, 3):
            myfile.write('Learning Rate {}, Perplexity {}\n'.format(lr, perp))
            myfile.write('{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}\n'
                         .format('','CAM2', 'INRIA', 'VOC', 'COCO', 'SUN', 'KITTI', 'Caltech'))

            tsne = TSNE(learning_rate=lr, random_state=0, perplexity=perp, n_iter=1500)
            coords = tsne.fit_transform(feats)

            cam2 = np.average(coords[0:100], axis=0).reshape(1, -1)
            inria = np.average(coords[100:200], axis=0).reshape(1, -1)
            voc = np.average(coords[200:300], axis=0).reshape(1, -1)
            coco = np.average(coords[300:400], axis=0).reshape(1, -1)
            sun = np.average(coords[400:500], axis=0).reshape(1, -1)
            kitti = np.average(coords[500:600], axis=0).reshape(1, -1)
            cal = np.average(coords[600:700], axis=0).reshape(1, -1)

            cam_inria = cdist(cam2, inria)[0][0]
            cam_voc = cdist(cam2, voc)[0][0]
            cam_coco = cdist(cam2, coco)[0][0]
            cam_sun = cdist(cam2, sun)[0][0]
            cam_kitti = cdist(cam2, kitti)[0][0]
            cam_cal = cdist(cam2, cal)[0][0]
            inria_voc = cdist(inria, voc)[0][0]
            inria_coco = cdist(inria, coco)[0][0]
            inria_sun = cdist(inria, sun)[0][0]
            inria_kitti = cdist(inria, kitti)[0][0]
            inria_cal = cdist(inria, cal)[0][0]
            voc_coco = cdist(voc, coco)[0][0]
            voc_sun = cdist(voc, sun)[0][0]
            voc_kitti = cdist(voc, kitti)[0][0]
            voc_cal = cdist(voc, cal)[0][0]
            coco_sun = cdist(coco, sun)[0][0]
            coco_kitti = cdist(coco, kitti)[0][0]
            coco_cal = cdist(coco, cal)[0][0]
            sun_kitti = cdist(sun, kitti)[0][0]
            sun_cal = cdist(sun, cal)[0][0]
            kitti_cal = cdist(kitti, cal)[0][0]

            cam_inrias.append(cam_inria)
            cam_vocs.append(cam_voc)
            cam_cocos.append(cam_coco)
            cam_suns.append(cam_sun)
            cam_kittis.append(cam_kitti)
            cam_cals.append(cam_cal)
            inria_vocs.append(inria_voc)
            inria_cocos.append(inria_coco)
            inria_suns.append(inria_sun)
            inria_kittis.append(inria_kitti)
            inria_cals.append(inria_cal)
            voc_cocos.append(voc_coco)
            voc_suns.append(voc_sun)
            voc_kittis.append(voc_kitti)
            voc_cals.append(voc_cal)
            coco_suns.append(coco_sun)
            coco_kittis.append(coco_kitti)
            coco_cals.append(coco_cal)
            sun_kittis.append(sun_kitti)
            sun_cals.append(sun_cal)
            kitti_cals.append(kitti_cal)

            myfile.write('{:<10}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}\n'
                         .format('CAM2', 0, cam_inria, cam_voc, cam_coco, cam_sun, cam_kitti, cam_cal))
            myfile.write('{:<10}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}\n'
                         .format('INRIA', cam_inria, 0, inria_voc, inria_coco, inria_sun, inria_kitti, inria_cal))
            myfile.write('{:<10}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}\n'
                         .format('VOC', cam_voc, inria_voc, 0, voc_coco, voc_sun, voc_kitti, voc_cal))
            myfile.write('{:<10}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}\n'
                         .format('COCO', cam_coco, inria_coco, voc_coco, 0, coco_sun, coco_kitti, coco_cal))
            myfile.write('{:<10}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}\n'
                         .format('SUN', cam_sun, inria_sun, voc_sun, coco_sun, 0, sun_kitti, sun_cal))
            myfile.write('{:<10}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}\n'
                         .format('KITTI', cam_kitti, inria_kitti, voc_kitti, coco_kitti, sun_kitti, 0, kitti_cal))
            myfile.write('{:<10}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}\n\n\n'
                         .format('Caltech', cam_cal, inria_cal, voc_cal, coco_cal, sun_cal, kitti_cal, 0))

cam_inria = np.average(np.asarray(cam_inrias))
cam_voc = np.average(np.asarray(cam_vocs))
cam_coco = np.average(np.asarray(cam_cocos))
cam_sun = np.average(np.asarray(cam_suns))
cam_kitti = np.average(np.asarray(cam_kittis))
cam_cal = np.average(np.asarray(cam_cals))
inria_voc = np.average(np.asarray(inria_vocs))
inria_coco = np.average(np.asarray(inria_cocos))
inria_sun = np.average(np.asarray(inria_suns))
inria_kitti = np.average(np.asarray(inria_kittis))
inria_cal = np.average(np.asarray(inria_cals))
voc_coco = np.average(np.asarray(voc_cocos))
voc_sun = np.average(np.asarray(voc_suns))
voc_kitti = np.average(np.asarray(voc_kittis))
voc_cal = np.average(np.asarray(voc_cals))
coco_sun = np.average(np.asarray(coco_suns))
coco_kitti = np.average(np.asarray(coco_kittis))
coco_cal = np.average(np.asarray(coco_cals))
sun_kitti = np.average(np.asarray(sun_kittis))
sun_cal = np.average(np.asarray(sun_cals))
kitti_cal = np.average(np.asarray(kitti_cals))

with open('avg.txt', 'w') as myfile:
    myfile.write('\n')
    myfile.write('{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}\n'
                 .format('', 'CAM2', 'INRIA', 'VOC', 'COCO', 'SUN', 'KITTI', 'Caltech'))
    myfile.write('{:<10}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}\n'
                 .format('CAM2', 0, cam_inria, cam_voc, cam_coco, cam_sun, cam_kitti, cam_cal))
    myfile.write('{:<10}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}\n'
                 .format('INRIA', cam_inria, 0, inria_voc, inria_coco, inria_sun, inria_kitti, inria_cal))
    myfile.write('{:<10}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}\n'
                 .format('VOC', cam_voc, inria_voc, 0, voc_coco, voc_sun, voc_kitti, voc_cal))
    myfile.write('{:<10}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}\n'
                 .format('COCO', cam_coco, inria_coco, voc_coco, 0, coco_sun, coco_kitti, coco_cal))
    myfile.write('{:<10}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}\n'
                 .format('SUN', cam_sun, inria_sun, voc_sun, coco_sun, 0, sun_kitti, sun_cal))
    myfile.write('{:<10}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}\n'
                 .format('KITTI', cam_kitti, inria_kitti, voc_kitti, coco_kitti, sun_kitti, 0, kitti_cal))
    myfile.write('{:<10}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}\n\n\n'
                 .format('Caltech', cam_cal, inria_cal, voc_cal, coco_cal, sun_cal, kitti_cal, 0))

