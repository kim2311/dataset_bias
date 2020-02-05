"""
authors: zkapach and fbordwel
"""

import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import itertools
import os
import glob
import time
import math
import subprocess
from random import shuffle
from sklearn.ensemble import BaggingClassifier
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from sklearn.metrics import confusion_matrix
import warnings
import random
warnings.filterwarnings('ignore')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True: # Call with two outputs if vis==True to visualize the HOG
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    else:      # Otherwise call with one output
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features

def classes_to_dict(classes):
    dict_classes = {}
    for i, name in enumerate(classes):
        dict_classes[name] = i
    return dict_classes

def switch_rows_cols(cm, classes, new_order):
    new_cm = np.copy(cm)
    for idx, nameA in enumerate(classes):
        for jdx, nameB in enumerate(classes):
            xVal = classes.index(new_order[idx])
            yVal = classes.index(new_order[jdx])
            new_cm[idx,jdx] = cm[xVal,yVal]
    # for i, name in enumerate(classes):
    #     new_cm[i,:] = cm[dict_classes[new_order[i]],:]

    # for i, name in enumerate(classes):
    #     new_cm[:,i] = cm[:, dict_classes[new_order[i]]]
    return new_cm

# Define a function to extract features from a list of images
def img_features(feature_image, feat_type, hist_bins, orient,
                        pix_per_cell, cell_per_block, hog_channel):
    file_features = []

    # Call get_hog_features() with vis=False, feature_vec=True
    if feat_type == 'gray':
        feature_image = cv2.resize(feature_image, (32,32))
        feature_image = cv2.cvtColor(feature_image, cv2.COLOR_BGR2GRAY)
        file_features.append(feature_image.ravel())
    elif feat_type == 'color':
        feature_image = cv2.resize(feature_image, (32,32))
        file_features.append(feature_image.ravel())
    elif feat_type == 'hog':
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel],
                                        orient, pix_per_cell, cell_per_block,
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
        else:
            feature_image = cv2.cvtColor(feature_image, cv2.COLOR_BGR2GRAY)
            feature_image = cv2.resize(feature_image, (128,256))
            hog_features = get_hog_features(feature_image[:,:], orient,
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)

            # Append the new feature vector to the features list
        file_features.append(hog_features)
    return file_features

def extract_features(imgs, feat_type, spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0):

    # Create a list to append feature vectors to
    features = []

    # Iterate through the list of images
    for file_p in imgs:
        file_features = []
        image = cv2.imread(file_p) # Read in each imageone by one
        if isinstance(image, np.ndarray) == True:
            feature_image = np.copy(image)
            file_features = img_features(feature_image, feat_type, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel)
            features.append(np.concatenate(file_features))
    return features # Return list of feature vectors

# LOAD IN DATA for Helps computer
#print('===== STARTING =====')

train_cam = '/local/b/cam2/data/cam2_new'
train_COCO = '/local/b/cam2/data/coco/images/train2014'
train_INRIA = '/local/b/cam2/data/INRIAPerson/Train'
train_caltech0 = '/local/b/cam2/data/caltech_pedestrian/extracted_data/set00/'
train_caltech1 = '/local/b/cam2/data/caltech_pedestrian/extracted_data/set01/'
train_caltech2 = '/local/b/cam2/data/caltech_pedestrian/extracted_data/set02/'
train_caltech3 = '/local/b/cam2/data/caltech_pedestrian/extracted_data/set03/'
train_caltech4 = '/local/b/cam2/data/caltech_pedestrian/extracted_data/set04/'
train_caltech5 = '/local/b/cam2/data/caltech_pedestrian/extracted_data/set05/'
test_caltech6 = '/local/b/cam2/data/caltech_pedestrian/extracted_data/set06/'
test_caltech7 = '/local/b/cam2/data/caltech_pedestrian/extracted_data/set07/'
test_caltech8 = '/local/b/cam2/data/caltech_pedestrian/extracted_data/set08/'
test_caltech9 = '/local/b/cam2/data/caltech_pedestrian/extracted_data/set09/'
test_caltech10 = '/local/b/cam2/data/caltech_pedestrian/extracted_data/set10/'
train_ImageNet = '/local/b/cam2/data/ILSVRC2012_Classification/train/'
train_Pascal = '/local/b/cam2/data/VOCdevkit/VOC2012/JPEGImages/'
train_Sun = '/local/b/cam2/data/SUN2012/Images/'
train_kitti = '/local/b/cam2/data/kitti/KITTIdevkit/KITTI2013/image_2/'
train_usps = '/local/b/cam2/data/usps/Numerals/'
train_mnist = '/local/b/cam2/data/mnist_new/mnist_png/training/'
train_mnistm = '/local/b/cam2/data/mnist_m/mnist_m_train/'
train_svhn = '/local/b/cam2/data/svhn/train/'

#for helps computer
caltech_train = (glob.glob(train_caltech0 + '/**/*.jpg', recursive = True)) + (glob.glob(train_caltech1 + '/**/*.jpg', recursive = True))+ (glob.glob(train_caltech2 + '/**/*.jpg', recursive = True))+ (glob.glob(train_caltech3 + '/**/*.jpg', recursive = True))+ (glob.glob(train_caltech4 + '/**/*.jpg', recursive = True))+ (glob.glob(train_caltech5 + '/**/*.jpg', recursive = True))#+ (glob.glob(train_caltech + '/**/*.bmp', recursive = True))
caltech_test = (glob.glob(test_caltech6 + '/**/*.jpg', recursive = True)) + (glob.glob(test_caltech7 + '/**/*.jpg', recursive = True))+ (glob.glob(test_caltech8 + '/**/*.jpg', recursive = True))+ (glob.glob(test_caltech9 + '/**/*.jpg', recursive = True))+ (glob.glob(test_caltech10 + '/**/*.jpg', recursive = True))#+ (glob.glob(train_caltech + '/**/*.bmp', recursive = True))
coco = glob.glob(train_COCO + '/**/*.png', recursive = True) + (glob.glob(train_COCO + '/**/*.jpg', recursive = True))#+ (glob.glob(train_COCO + '/**/*.bmp', recursive = True))
imagenet = glob.glob(train_ImageNet + '/**/*.JPEG', recursive = True) + (glob.glob(train_ImageNet + '/**/*.jpg', recursive = True))#+ (glob.glob(train_ImageNet + '/**/*.bmp', recursive = True))
pascal = glob.glob(train_Pascal + '/**/*.png', recursive = True) + (glob.glob(train_Pascal + '/**/*.jpg', recursive = True))#+ (glob.glob(train_Pascal + '/**/*.bmp', recursive = True))
cam2 = glob.glob(train_cam + '/**/*.jpg', recursive = True)
inria = glob.glob(train_INRIA + '/**/*.png', recursive = True) + (glob.glob(train_INRIA + '/**/*.jpg', recursive = True))
sun = glob.glob(train_Sun + '/**/*.jpg', recursive = True)
kitti = glob.glob(train_kitti + '/**/*.png', recursive = True)
usps = glob.glob(train_usps + '/**/*.png', recursive = True) + (glob.glob(train_usps + '/**/*.jpg', recursive = True))
mnist = glob.glob(train_mnist + '/**/*.png', recursive = True) + (glob.glob(train_mnist + '/**/*.jpg', recursive = True))
mnistm = glob.glob(train_mnistm + '/**/*.png', recursive = True) + (glob.glob(train_mnistm + '/**/*.jpg', recursive = True))
svhn = glob.glob(train_svhn + '/**/*.png', recursive = True) + (glob.glob(train_svhn + '/**/*.jpg', recursive = True))
model = ['hog']
results = []

i = 0
for j in model:
    t3 = time.time()
    shuffle(coco)
    shuffle(caltech_train)
    shuffle(caltech_test)
    shuffle(imagenet)
    shuffle(pascal)
    shuffle(cam2)
    shuffle(inria)
    shuffle(sun)
    shuffle(kitti)
    shuffle(usps)
    shuffle(mnist)
    shuffle(mnistm)
    shuffle(svhn)
    ''' 
    print('len of coco', len(coco))
    print('len of caltech_test', len(caltech_test))
    print('len of caltech_train', len(caltech_train))
    '''
    print('len of imagenet', len(imagenet))
    '''
    print('len of pascal' ,len(pascal))
    print('len of cam2', len(cam2))
    print('len of inria', len(inria))
    print('len of sun', len(sun))
    print('len of kitti', len(kitti))
    print('len of usps', len(usps))
    print('len of mnist', len(mnist))
    print('len of mnistm', len(mnistm))
    print('len of svhn', len(svhn))
    '''
    
    # Specify size of training and testing sets
    dataset_size_train = 500
    dataset_size_test = 500
    dataset_total = dataset_size_train+dataset_size_test

    caltech_train = caltech_train[0:dataset_size_train]
    caltech_test = caltech_test[0:dataset_size_test]
    coco1 = coco[0:(dataset_size_train+dataset_size_test)]
    imagenet1 = imagenet[0:(dataset_size_train+dataset_size_test)]
    pascal1 = pascal[0:(dataset_size_train+dataset_size_test)]
    cam21 = cam2[0:(dataset_size_train+dataset_size_test)]
    inria1 = inria[0:(dataset_size_train+dataset_size_test)]
    sun1 = sun[0:(dataset_size_train+dataset_size_test)]
    kitti1 = kitti[0:(dataset_size_train+dataset_size_test)]
    usps1 = usps[0:(dataset_size_train+dataset_size_test)]
    mnist1 = mnist[0:(dataset_size_train+dataset_size_test)]
    mnistm1 = mnistm[0:(dataset_size_train+dataset_size_test)]
    svhn1 = svhn[0:(dataset_size_train+dataset_size_test)]
    '''
    print('len of coco', len(coco1))
    print('len of caltech_test', len(caltech_test))
    print('len of caltech_train', len(caltech_train))
    print('len of imagenet', len(imagenet1))
    print('len of pascal' ,len(pascal1))
    print('len of cam2', len(cam21))
    print('len of inria', len(inria1))
    print('len of sun', len(sun1))
    print('len of kitti', len(kitti1))
    print('len of usps', len(usps1))
    print('len of mnist', len(mnist1))
    print('len of mnistm', len(mnistm1))
    print('len of svhn', len(svhn1))
    '''
    orient = 8  # HOG orientations
    pix_per_cell = 8 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = 0 # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16) # Spatial binning dimensions
    hist_bins = 32    # Number of histogram bins

    coco_feat = extract_features(coco1, j,spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
    caltech_train_feat = extract_features(caltech_train, j,spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
    caltech_test_feat = extract_features(caltech_test, j,spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
    imagenet_feat = extract_features(imagenet1, j,spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
    pascal_feat = extract_features(pascal1, j,spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
    cam2_feat = extract_features(cam21, j,spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
    inria_feat = extract_features(inria1, j,spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
    sun_feat = extract_features(sun1, j,spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
    kitti_feat = extract_features(kitti1, j,spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
    usps_feat = extract_features(usps1, j,spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
    mnist_feat = extract_features(mnist1, j,spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
    mnistm_feat = extract_features(mnistm1, j,spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
    svhn_feat = extract_features(svhn1, j,spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
    
    '''
    print('===== LENGTH AFTER CUT =====')
    print('len of coco', len(coco1))
    print('len of caltech_train', len(caltech_train))
    print('len of caltech_test', len(caltech_test))
    print('len of imagenet', len(imagenet1))
    print('len of pascal' ,len(pascal1))
    print('len of cam2', len(cam21))
    print('len of inria', len(inria1))
    print('len of sun', len(sun1))
    print('len of kitti', len(kitti1))
    '''

    coco_train = coco_feat[0:dataset_size_train]
    coco_test = coco_feat[dataset_size_train:(dataset_size_train+dataset_size_test)]
    caltech_train = caltech_train_feat
    caltech_test = caltech_test_feat
    imagenet_train = imagenet_feat[0:dataset_size_train]
    imagenet_test = imagenet_feat[dataset_size_train:(dataset_size_train+dataset_size_test)]
    pascal_train = pascal_feat[0:dataset_size_train]
    pascal_test = pascal_feat[dataset_size_train:(dataset_size_train+dataset_size_test)]
    cam2_train = cam2_feat[0:dataset_size_train]
    cam2_test = cam2_feat[dataset_size_train:(dataset_size_train+dataset_size_test)]
    inria_train = inria_feat[0:dataset_size_train]
    inria_test = inria_feat[dataset_size_train:(dataset_size_train+dataset_size_test)]
    sun_train = sun_feat[0:dataset_size_train]
    sun_test = sun_feat[dataset_size_train:(dataset_size_train+dataset_size_test)]
    kitti_train = kitti_feat[0:dataset_size_train]
    kitti_test = kitti_feat[dataset_size_train:(dataset_size_train+dataset_size_test)]
    usps_train = usps_feat[0:dataset_size_train]
    usps_test = usps_feat[dataset_size_train:(dataset_size_train+dataset_size_test)]
    mnist_train = mnist_feat[0:dataset_size_train]
    mnist_test = mnist_feat[dataset_size_train:(dataset_size_train+dataset_size_test)]
    mnistm_train = mnistm_feat[0:dataset_size_train]
    mnistm_test = mnistm_feat[dataset_size_train:(dataset_size_train+dataset_size_test)]
    svhn_train = svhn_feat[0:dataset_size_train]
    svhn_test = svhn_feat[dataset_size_train:(dataset_size_train+dataset_size_test)]

    """
    print('===== LENGTH AFTER FEATURE EXTRACTION AND CUT =====')
    print('len of coco', len(coco_train), len(coco_test))
    print('len of caltech', len(caltech_train), len(caltech_test))
    print('len of imagenet', len(imagenet_train), len(imagenet_test))
    print('len of pascal' ,len(pascal_train), len(pascal_test))
    print('len of cam2', len(cam2_train), len(cam2_test))
    print('len of inria', len(inria_train), len(inria_test))
    print('len of sun', len(sun_train), len(sun_test))
    print('len of kitti', len(kitti_train), len(kitti_test))
    """
    X_train_m = np.vstack((coco_train, caltech_train, imagenet_train, pascal_train, cam2_train, inria_train, sun_train, kitti_train, usps_train, mnist_train, mnistm_train, svhn_train)).astype(np.float64) #, mix_train)).astype(np.float64)
    X_test_m = np.vstack((coco_test, caltech_test, imagenet_test, pascal_test, cam2_test, inria_test, sun_test, kitti_test, usps_test, mnist_test, mnistm_test, svhn_test)).astype(np.float64) #, mix_test)).astype(np.float64)
    X_train_scaler = StandardScaler().fit(X_train_m)
    X_test_scaler = StandardScaler().fit(X_test_m)

    X_train_scaled = X_train_scaler.transform(X_train_m)
    X_test_scaled = X_test_scaler.transform(X_test_m)
    y_train = np.hstack((np.ones(len(coco_train)), np.full(len(caltech_train), 2), np.full(len(imagenet_train), 3), np.full(len(pascal_train), 4), np.full(len(cam2_train), 5), np.full(len(inria_train), 6), np.full(len(sun_train), 7), np.full(len(kitti_train), 8), np.full(len(usps_train), 9), np.full(len(mnist_train), 10), np.full(len(mnistm_train), 11), np.full(len(svhn_train), 12))) #, np.full(len(mix_train), 9)))
    y_test = np.hstack((np.ones(len(coco_test)), np.full(len(caltech_test), 2), np.full(len(imagenet_test), 3), np.full(len(pascal_test), 4), np.full(len(cam2_test), 5), np.full(len(inria_test), 6), np.full(len(sun_test), 7), np.full(len(kitti_test), 8), np.full(len(usps_test), 9), np.full(len(mnist_test), 10), np.full(len(mnistm_test), 11), np.full(len(svhn_test), 12))) #, np.full(len(mix_test), 9)))
    """
    print('Using:',orient,'orientations', pix_per_cell, 'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train_scaled[0]))
    """
    # Begin training
    svc = LinearSVC(loss='hinge', multi_class = 'ovr')
    t=time.time()

   # print('Start training')
    model_fit = svc.fit(X_train_scaled, y_train)
    t2 = time.time()

    #print(round(t2-t, 2), 'Seconds to train SVC...')
    #print('Test Accuracy of SVC = ', round(model_fit.score(X_test_scaled, y_test), 4)) # Check the score of the SVC
    results.append(round(model_fit.score(X_test_scaled, y_test), 4))

    t4 = time.time()
    #print(round(t4-t3, 2), 'Seconds to run')

    # Begin confusion matrix
    if i == 0:
        y_pred = model_fit.predict(X_test_scaled)

        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_test, y_pred)
        np.set_printoptions(precision=2)

        # Plot normalized confusion matrix
        class_names = ('COCO', 'Caltech', 'ImageNet', 'Pascal', 'Cam2', 'INRIA', 'Sun', 'KITTI', 'USPS', 'MNIST', 'MNISTM', 'SVHN')
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')

        plt.show()

#print('results are', results)
