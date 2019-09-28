import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
%matplotlib inline  

bs = 1 #batch size

mnist = MNIST('./', train=False, transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))
                                        ]), download=True)

mnist_loader = torch.utils.data.DataLoader(
        mnist,
        batch_size=bs, shuffle=True)

mnist_imgs = []
mnist_labs = []

for i, (img, lab) in enumerate(mnist_loader):
    if i>=0 and i < 1000:
        mnist_imgs.append(img)
        mnist_labs.append(lab)
        
mnist_imgs = torch.stack(mnist_imgs).squeeze(dim=0)
mnist_labs = torch.stack(mnist_labs).squeeze(dim=0)
mnist_imgs = mnist_imgs.view(mnist_imgs.shape[0], -1)

m = TSNE(learning_rate=100, random_state=0, perplexity=100)
mnist_2D = m.fit_transform(mnist_imgs)

colors = ['b', 'g', 'r', 'c', 'm', 'k', 'w', 'y', '0.3', '0.7']
labs = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

for i in range(1000):
#     col = '{}'.format(float(mnist_labs[i]) / 10.0)
    _ = plt.scatter(mnist_2D[i, 0], mnist_2D[i, 1], marker='.', color=colors[int(mnist_labs[i])])

_ = plt.title('MNIST classes')
_ = plt.xlabel('Feature A')
_ = plt.ylabel('Feature B')
_ = plt.legend(['MNIST'], loc='best')

plt.show()