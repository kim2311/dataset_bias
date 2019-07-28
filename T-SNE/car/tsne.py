from sklearn.manifold import TSNE
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable

model = models.resnet18(pretrained=True)
layer = model._modules.get('avgpool')
model.eval()
scaler = transforms.Scale((224, 224))
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
to_tensor = transforms.ToTensor()

LABELS = ['ImageNet', 'VOC', 'COCO', 'SUN', 'KITTI']
MARKER = ['v', 'd', '*', 'x', '+']
COLOR = ['tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:cyan']


def get_vector(path):
    img = Image.open(path)
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    my_embedding = torch.zeros((1,512,1,1))

    def copy_data(m, i, o):
        my_embedding.copy_(o.data)
    h = layer.register_forward_hook(copy_data)
    model(t_img)
    h.remove()
    return my_embedding


def get_normalized(path):
    img = Image.open(path)
    img = img.resize((224, 224)).convert('RGB')
    img = normalize(to_tensor(img))
    print(torch.cat((img, img)).shape)
    return img


def get_img_feat(path):
    imgs = []
    feats = []
    for file in os.listdir(path):
        if file.endswith('.jpg') or file.endswith('.JPEG'):
            img = get_normalized(os.path.join(path, file))
            feat = get_vector(os.path.join(path, file))
            img = np.asarray(img).reshape((-1))
            feat = np.asarray(feat).reshape((-1))
            imgs.append(img)
            feats.append(feat)
    imgs = np.asarray(imgs)
    np.save(path, imgs)
    feats = np.asarray(feats)
    labels = [path] * 100
    print(imgs.shape)
    print(feats.shape)
    return imgs, feats, labels


imgs, feats, labels = get_img_feat('ImageNet')

for dataset in LABELS:
    if dataset is not 'ImageNet':
        d_imgs, d_feats, d_labels = get_img_feat(dataset)
        imgs = np.concatenate((imgs, d_imgs))
        feats = np.concatenate((feats, d_feats))
        labels += d_labels

tsne = TSNE(learning_rate=20, random_state=0, perplexity=21, n_iter=1500)
save_path_co = 'coords.npy'
coord = tsne.fit_transform(feats)
np.save(save_path_co, coord)


'''
for lr in range(10, 301, 5):
    for perp in range(5, 51, 2):
        #save_path_o = 'plots/normalized/lr{0}_pp{1}_niter1500.jpg'.format(lr, perp)
        save_path_f = 'new_plots/lr{0}_pp{1}_niter1500.jpg'.format(lr, perp)
        filename = 'new_coords/lr{0}_pp{1}_rs0_niter1500.txt'.format(lr, perp)
        #tsne = TSNE(learning_rate=lr, random_state=0, perplexity=perp, n_iter=1500)
        #for i in range(6):
        #    coord = tsne.fit_transform(imgs[i])
        #    plt.scatter(coord[:, 0], coord[:, 1], color=COLOR[i], marker=MARKER[i], label=LABELS[i])
        #plt.legend()
        #plt.savefig(save_path_o)
        #plt.close()


        tsne = TSNE(learning_rate=lr, random_state=0, perplexity=perp, n_iter=1500)
        coord = tsne.fit_transform(feats)
        for i in range(5):
            plt.scatter(coord[i*100:i*100+100, 0], coord[i*100:i*100+100, 1], color=COLOR[i], marker=MARKER[i], label=LABELS[i])
        plt.legend()
        plt.savefig(save_path_f)
        plt.close()
        with open(filename, 'w') as myfile:
            for i in range(500):
                myfile.write('{:<30}{}'.format(coord[i, 0], coord[i, 1]))
                myfile.write('\n')
'''
