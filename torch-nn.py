#!/usr/bin/env python

# Copyright (C) 2021-2025 Simon Crase

# This is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with GNU Emacs.  If not, see <http://www.gnu.org/licenses/>


'''
    Neural Net with Pytorch based on
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
'''

from argparse import ArgumentParser
from os.path import splitext,join
from pathlib import Path
from time import time
from matplotlib.pyplot import figure, show
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Normalization:
    '''
    This class is reponsible for normalizing images to meet the needs
    of pytorch and matplotlib
    '''
    mean = 0.5
    std = 0.5
    @staticmethod
    def create_transform():
        '''
        Create a chains of image transformations to be applied to data,
        typically images, before they are used in a PyTorch model.

        The first transformation converts a PIL Image or NumPy ndarray into a PyTorch FloatTensor.
        It also scales the pixel values from the range [0, 255] to [0.0, 1.0].
        For PIL Images, it converts the image from (H, W, C) format (Height, Width, Channels) to
        (C, H, W) format, which is the standard tensor format for PyTorch.

        The second transformation normalizes a tensor image into the range [-1,1].
        '''
        Means = (Normalization.mean,Normalization.mean,Normalization.mean)
        Stds = (Normalization.std,Normalization.std,Normalization.std)
        return transforms.Compose([transforms.ToTensor(),transforms.Normalize(Means,Stds)])

    @staticmethod
    def unnormalize(img):
        '''
        bring image back to the 0-1 range, which is what matplotlib expects
        '''
        return Normalization.std*img + Normalization.mean

def imshow(img,ax=None):
    '''
    Visualize image

    Parameters:
        img    Represents the normalized image (in the range -1 to 1),
    '''
    npimg = Normalization.unnormalize(img).numpy()
    ax.imshow(np.transpose(npimg, (1, 2, 0)))


class Net(nn.Module):
    '''
    A neural network with 2 convolutional layers, feeding into 3 linear layers
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84,out_features =10)

    def forward(self, x): # shape [4, 3, 32, 32]
        x = self.pool(F.relu(self.conv1(x)))  # shape [4, 6, 14, 14].
        x = self.pool(F.relu(self.conv2(x))) # shape [4, 16, 5, 5]
        x = x.view(-1, 16 * 5 * 5) # shape [4, 400]
        x = F.relu(self.fc1(x))  # shape [4, 120]
        x = F.relu(self.fc2(x))  # shape [4, 84]
        x = self.fc3(x) # shape [4, 10]
        return x

def get_filename(name=Path(__file__).stem):
    '''
    Determine file name for plots
    '''
    filename, file_extension = splitext(name)
    if len(file_extension) == 0:
        filename = f'{filename}.png'
    return filename


def train(root='./data', show=False, lr=0.001, momentum=0.9, n=1000, freq=20, plot='./train.png',
          PATH='./cifar_net.pth', figs='./figs',transform=None,m=None):
    '''
    Train network against CIFAR10 data

    Parameters:
        root        Path to folder where data are stored
        show        Controls whether plots are to be displayed
        lr          Learning factor
        momentum    Momentum factor for training
        n           Number of epochs for training
        m           Number of samples for training (omit for all samples)
        freq        Frequency for reporting (number of samples)
        plot        Name of plot file
        PATH        Path to file where neural net weights are stored
        figs        Location for storing plot files
        transform   Used to transform images for format expected by PyTorch
    '''
    trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)

    fig = figure(figsize=(10, 10))
    # display some random training images
    dataiter = iter(trainloader)
    images, _ = next(dataiter)
    imshow(torchvision.utils.make_grid(images),ax = fig.add_subplot(2, 1, 1))

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    losses = []

    for epoch in range(n):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data  # shape inputs [4,3,32,32]
            optimizer.zero_grad()
            outputs = net(inputs)  # shape outputs [4,10]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i > freq and i % freq == 1:
                mean_loss = running_loss / freq
                print(f'Epoch = {epoch}, seq = {i}, mean_loss = {mean_loss}')
                losses.append(mean_loss)
                running_loss = 0.0
                if m != None and i >= m:
                    break

    print('Finished Training')
    torch.save(net.state_dict(), PATH)

    ax = fig.add_subplot(2, 1, 2)
    ax.plot(losses,label=f'Loss: n={n}, lr={lr}, momentum={momentum}')
    ax.set_ylim(bottom=0)
    ax.set_title('Training ')

    fig.savefig(join(figs,get_filename(plot)))


def test(root='./data', PATH='./cifar_net.pth',transform=None):
    '''
    Train network against CIFAR10 data

    Parameters:
        root        Path to folder where data are stored
        PATH        Path to file where neural net weights are stored
        transform   Used to transform images for format expected by PyTorch
    '''
    testset = torchvision.datasets.CIFAR10(root=root,
                                           train=False,
                                           download=True,
                                           transform=transform)

    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=4,
                                             shuffle=False,
                                             num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    fig = figure(figsize=(10, 10))
    imshow(torchvision.utils.make_grid(images),ax=fig.add_subplot(1,1,1))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))   # Why 4?

    net = Net()
    net.load_state_dict(torch.load(PATH))
    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    class_correct = list(0. for _ in range(len(classes)))
    class_total = list(0. for _ in range(len(classes)))

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(len(classes)):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=['train', 'test'], default='train',help='train or test')
    parser.add_argument('--n', default=10, type=int, help='Number of epochs for training')
    parser.add_argument('--m', default=None, type=int, help='Number of samples for training (omit for all samples)')
    parser.add_argument('--freq', default=100, type=int, help='Frequency for reporting (number of samples)')
    parser.add_argument('--lr', default=0.001, type=float, help = 'Learning rate')
    parser.add_argument('--momentum', default=0.9, type=int, help='Momentum factor for training')
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--PATH', default='./cifar_net.pth', help='Path to file where neural net weights are stored')
    parser.add_argument('--root', default='./data', help='Path to folder where data are stored')
    parser.add_argument('--plot', default=Path(__file__).stem,help='Name of plot file')
    parser.add_argument('--figs', default='./figs', help='Location for storing plot files')
    return parser.parse_args()


if __name__ == '__main__':
    start  = time()
    args = parse_args()

    match args.action:
        case 'train':
            train(root=args.root, show=args.show, lr=args.lr, momentum=args.momentum, n=args.n,
                  freq=args.freq, plot=args.plot, PATH=args.PATH,figs=args.figs,
                  transform=Normalization.create_transform(),m=args.m)

        case 'test':
            test(root=args.root, PATH=args.PATH, transform=Normalization.create_transform())

    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')

    if args.show:
        show()
