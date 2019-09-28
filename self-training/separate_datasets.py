import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
%matplotlib inline  

bs = 1000 #batch size

cifar = CIFAR10('./', train=False, transform=transforms.Compose([
                                            transforms.CenterCrop(28),
                                            transforms.Grayscale(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))
                                        ]), download=True)

mnist = MNIST('./', train=False, transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))
                                        ]), download=True)

mnist_loader = torch.utils.data.DataLoader(
        mnist,
        batch_size=bs, shuffle=True)

cifar_loader = torch.utils.data.DataLoader(
        cifar,
        batch_size=bs, shuffle=True)

mnist_imgs = []
cifar_imgs = []

for i, (img, lab) in enumerate(mnist_loader):
    if i==0:
        mnist_imgs.append(img)
        
for i, (img, lab) in enumerate(cifar_loader):
    if i==0:
        cifar_imgs.append(img)
        
mnist_imgs = torch.stack(mnist_imgs).squeeze(dim=0)
cifar_imgs = torch.stack(cifar_imgs).squeeze(dim=0)

mnist_imgs = mnist_imgs.view(mnist_imgs.shape[0], -1)
cifar_imgs = cifar_imgs.view(cifar_imgs.shape[0], -1)

m = TSNE(learning_rate=50, random_state=0)
mnist_2D = m.fit_transform(mnist_imgs)
cifar_2D = m.fit_transform(cifar_imgs)

_ = plt.plot(mnist_2D[:, 0], mnist_2D[:, 1], marker='.', linestyle='none', color='blue')
_ = plt.plot(cifar_2D[:, 0], cifar_2D[:, 1], marker='.', linestyle='none', color='red')
_ = plt.title('MNIST and CIFAR-10 visualized')
_ = plt.xlabel('Feature A')
_ = plt.ylabel('Feature B')
_ = plt.legend(['MNIST', 'CIFAR'], loc='best')

plt.show()