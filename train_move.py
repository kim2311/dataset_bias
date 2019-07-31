from __future__ import print_function
import argparse
import torch
import os
import shutil
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torchvision import datasets, transforms
from datasets import MNISTM

# This is the architecture to train
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        # print(x.shape)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

#sigmoid instead of softmax layer for testing. Helps get class confidences and allows multiclass classifications
class OpenNet(nn.Module):
    def __init__(self):
        super(OpenNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        # print(x.shape)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.sigmoid(x)
    
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    tot_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('TRAIN: epoch num: {} loss: {}'.format(epoch, tot_loss))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    conf = 0.0
    corr_conf = 0.0
    incorr_conf = 0.0
    incorr_len = 0
    corr_len = 0

    conf_thres = 0.98
    above = 0
    below = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            pred_values, _ = output.max(dim=1, keepdim=True) # get the values of the max log-probability

            corr = pred.eq(target.view_as(pred)) #get the indices of the correct predictions as bool [test_batch_size, 1] tensor

            conf += torch.sum(pred_values) #sum up the total confidence          
            corr_conf += torch.sum(pred_values[corr == 1]) #sum up confidence for correct predictions           
            incorr_conf += torch.sum(pred_values[corr == 0]) #sum up confidence for incorrect predictions

            corr_len += len(pred_values[corr == 1]) #add the number of correct predictions
            incorr_len += len(pred_values[corr == 0]) #add the number of incorrect predictions

    # print('\nAccuracy for confidence threshold of {} is: {}\n'.format(conf_thres, above * 100 / (above + below)))
    test_loss /= len(test_loader.dataset)
    incorr_conf /= incorr_len
    corr_conf /= corr_len
    conf /= len(test_loader.dataset)

    print('\nTest set: Average confidence for all predictions: {}\n'.format(conf))
    print('\nTest set: Average confidence for correct predictions: {}\n'.format(corr_conf))
    print('\nTest set: Average confidence for incorrect predictions: {}\n'.format(incorr_conf))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--query_thres', type=float, default=99.999,
                        help='confidence threshold for movinf images from training to testing')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    model = Net().to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    train_set = datasets.MNIST('../data', train=True, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))
                                        ]))

    test_set = MNISTM(root="data", train=False, download=True)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # model.load_state_dict(torch.load('models/mnist_cnn.pt'))
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(),"models/testing.pt")
        
if __name__ == '__main__':
    main()