"""
authors: zkapach and fbordwel
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
import itertools
import glob
import time
from random import shuffle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.metrics import confusion_matrix
import warnings
from sklearn.model_selection import KFold
from pprint import pprint as pp
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


print('===== Starting Program  =====')

train_COCO = '/local/b/cam2/data/person/COCO/'
train_ImageNet = '/local/b/cam2/data/person/KITTI/'
train_Pascal = '/local/b/cam2/data/person/VOC/'
train_Sun = '/local/b/cam2/data/person/SUN/'

# for helps computer
coco = glob.glob(train_COCO + '/**/*.png', recursive = True) + (glob.glob(train_COCO + '/**/*.jpg', recursive = True))#+ (glob.glob(train_COCO + '/**/*.bmp', recursive = True))
imagenet = glob.glob(train_ImageNet + '/**/*.JPEG', recursive = True) + (glob.glob(train_ImageNet + '/**/*.jpg', recursive = True))#+ (glob.glob(train_ImageNet + '/**/*.bmp', recursive = True))
pascal = glob.glob(train_Pascal + '/**/*.png', recursive = True) + (glob.glob(train_Pascal + '/**/*.jpg', recursive = True))#+ (glob.glob(train_Pascal + '/**/*.bmp', recursive = True))
sun = glob.glob(train_Sun + '/**/*.jpg', recursive = True)
model = ['hog']
results = []

for j in model:
    t3 = time.time()
    shuffle(coco)
    shuffle(imagenet)
    shuffle(pascal)
    shuffle(sun)
    
    # Specify size of training and testing sets
    dataset_size_train = 900
    dataset_size_test = 100
    dataset_total = dataset_size_train + dataset_size_test

    coco1 = coco[0:(dataset_size_train + dataset_size_test)]
    imagenet1 = imagenet[0:(dataset_size_train + dataset_size_test)]
    pascal1 = pascal[0:(dataset_size_train + dataset_size_test)]
    sun1 = sun[0:(dataset_size_train + dataset_size_test)]
    
    sample_list = coco1 + imagenet1 + pascal1 + sun1
    #pp(sample_list)

    print('Length of COCO\t\t ', len(coco1))
    print('Length of ImageNet\t ', len(imagenet1))
    print('Length of VOC\t\t ' , len(pascal1))
    print('Length of SUN\t\t ', len(sun1))
    
    orient = 8  # HOG orientations
    pix_per_cell = 8  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block
    hog_channel = 0  # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16)  # Spatial binning dimensions
    hist_bins = 32  # Number of histogram bins
    
    coco_feat = extract_features(coco1, j,spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
    imagenet_feat = extract_features(imagenet1, j,spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
    pascal_feat = extract_features(pascal1, j,spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
    sun_feat = extract_features(sun1, j,spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)

    kfold = KFold(10, False, 1)
    
    data0 = np.array(coco_feat)
    data1 = np.array(imagenet_feat)
    data2 = np.array(pascal_feat)
    data3 = np.array(sun_feat)

    feature_list = coco_feat + imagenet_feat + pascal_feat + sun_feat
    dictionary = dict(zip(sample_list, feature_list))

    # Define wholesome datasets
    coco_wholesome = set()
    imagenet_wholesome = set()
    voc_wholesome = set()
    sun_wholesome = set()

    coco_self = set()
    imagenet_self = set()
    voc_self = set()
    sun_self = set()

    for kfold0, kfold1, kfold2, kfold3, x in zip(kfold.split(data0), kfold.split(data1), kfold.split(data2), kfold.split(data3), range(10)):

        print(f"== Iteration {x+1} ==")

        train0, test0 = kfold0
        train1, test1 = kfold1
        train2, test2 = kfold2
        train3, test3 = kfold3

        coco_train = data0[train0]
        coco_test = data0[test0]
        imagenet_train = data1[train1]
        imagenet_test = data1[test1]
        pascal_train = data2[train2]
        pascal_test = data2[test2]
        sun_train = data3[train3]
        sun_test = data3[test3]

        X_train_m = np.vstack((coco_train, imagenet_train, pascal_train, sun_train)).astype(np.float64)
        X_test_m = np.vstack((coco_test, imagenet_test, pascal_test, sun_test)).astype(np.float64)
        X_train_scaler = StandardScaler().fit(X_train_m)
        X_test_scaler = StandardScaler().fit(X_test_m)

        X_train_scaled = X_train_scaler.transform(X_train_m)
        X_test_scaled = X_test_scaler.transform(X_test_m)
        y_train = np.hstack((np.ones(len(coco_train)), np.full(len(imagenet_train), 2), np.full(len(pascal_train), 3), np.full(len(sun_train), 4)))
        y_test = np.hstack((np.ones(len(coco_test)), np.full(len(imagenet_test), 2), np.full(len(pascal_test), 3), np.full(len(sun_test), 4)))

        # print('Using:',orient,'orientations', pix_per_cell, 'pixels per cell and', cell_per_block,'cells per block')
        # print('Feature vector length:', len(X_train_scaled[0]))

        # Begin training
        svc = LinearSVC(loss='hinge', multi_class = 'ovr')
        t=time.time()

        print('Starting training...')
        model_fit = svc.fit(X_train_scaled, y_train)
        t2 = time.time()

        print(round(t2-t, 2), 'seconds to train SVC.')
        print('Test Accuracy of SVC = ', round(model_fit.score(X_test_scaled, y_test), 4))

        results.append(round(model_fit.score(X_test_scaled, y_test), 4))

        # Plot confusion matrix
        y_pred = model_fit.predict(X_test_scaled)
        # print(f"y_pred = {y_pred}")

        cnf_matrix = confusion_matrix(y_test, y_pred)
        np.set_printoptions(precision=2)

        class_names = ('COCO', 'ImageNet', 'Pascal', 'Sun')
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')

        plt.show()

        # Add samples to wholesome datasets
        coco_pred = y_pred[0:dataset_size_test]
        imagenet_pred = y_pred[dataset_size_test:(2*dataset_size_test)]
        voc_pred = y_pred[(2*dataset_size_test):(3*dataset_size_test)]
        sun_pred = y_pred[(3*dataset_size_test):(4*dataset_size_test)]

        ind = -1
        for sample in coco_pred:
            ind += 1
            if sample == 1:
                for key, value in dictionary.items():
                    if np.array_equal(value, coco_test[ind]):
                        coco_self.add(key)
            elif sample == 2:
                for key, value in dictionary.items():
                    if np.array_equal(value, coco_test[ind]):
                        imagenet_wholesome.add(key)
            elif sample == 3:
                for key, value in dictionary.items():
                    if np.array_equal(value, coco_test[ind]):
                        voc_wholesome.add(key)
            elif sample == 4:
                for key, value in dictionary.items():
                    if np.array_equal(value, coco_test[ind]):
                        sun_wholesome.add(key)

        ind = -1
        for sample in imagenet_pred:
            ind += 1
            if sample == 1:
                for key, value in dictionary.items():
                    if np.array_equal(value, imagenet_test[ind]):
                        coco_wholesome.add(key)
            elif sample == 2:
                for key, value in dictionary.items():
                    if np.array_equal(value, imagenet_test[ind]):
                        imagenet_self.add(key)
            elif sample == 3:
                for key, value in dictionary.items():
                    if np.array_equal(value, imagenet_test[ind]):
                        voc_wholesome.add(key)
            elif sample == 4:
                for key, value in dictionary.items():
                    if np.array_equal(value, imagenet_test[ind]):
                        sun_wholesome.add(key)

        ind = -1
        for sample in voc_pred:
            ind += 1
            if sample == 1:
                for key, value in dictionary.items():
                    if np.array_equal(value, pascal_test[ind]):
                        coco_wholesome.add(key)
            elif sample == 2:
                for key, value in dictionary.items():
                    if np.array_equal(value, pascal_test[ind]):
                        imagenet_wholesome.add(key)
            elif sample == 3:
                for key, value in dictionary.items():
                    if np.array_equal(value, pascal_test[ind]):
                        voc_self.add(key)
            elif sample == 4:
                for key, value in dictionary.items():
                    if np.array_equal(value, pascal_test[ind]):
                        sun_wholesome.add(key)

        ind = -1
        for sample in sun_pred:
            ind += 1
            if sample == 1:
                for key, value in dictionary.items():
                    if np.array_equal(value, sun_test[ind]):
                        coco_wholesome.add(key)
            elif sample == 2:
                for key, value in dictionary.items():
                    if np.array_equal(value, sun_test[ind]):
                        imagenet_wholesome.add(key)
            elif sample == 3:
                for key, value in dictionary.items():
                    if np.array_equal(value, sun_test[ind]):
                        voc_wholesome.add(key)
            elif sample == 4:
                for key, value in dictionary.items():
                    if np.array_equal(value, sun_test[ind]):
                        voc_self.add(key)

    with open('coco_wholesome.txt','w') as file:
        for item in coco_wholesome:
            #print(item)
            file.write("%s\n" % item)

    with open('kitti_wholesome.txt','w') as file:
        for item in imagenet_wholesome:
            file.write("%s\n" % item)

    with open('voc_wholesome.txt','w') as file:
        for item in voc_wholesome:
            file.write("%s\n" % item)

    with open('sun_wholesome.txt','w') as file:
        for item in sun_wholesome:
            file.write("%s\n" % item)

    with open('coco_self.txt','w') as file:
        for item in coco_self:
            #print(item)
            file.write("%s\n" % item)

    with open('kitti_self.txt','w') as file:
        for item in imagenet_self:
            file.write("%s\n" % item)

    with open('voc_self.txt','w') as file:
        for item in voc_self:
            file.write("%s\n" % item)

    with open('sun_self.txt','w') as file:
        for item in sun_self:
            file.write("%s\n" % item)

t4 = time.time()
print(round(t4-t3, 2), 'seconds to run experiment.')
print('Experiment finished.')
