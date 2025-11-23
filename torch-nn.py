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
from time import time
from matplotlib.pyplot import figure, show
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def imshow(img,ax=None):
    '''
    Visualize image

    Parameters:
        img    The represents the normalized image (in the range -1 to 1),
    '''
    # bring image back to the 0-1 range, which is what matplotlib expects
    img = img / 2 + 0.5
    npimg = img.numpy()
    ax.imshow(np.transpose(npimg, (1, 2, 0)))


class Net(nn.Module):
    '''
    A neural network with 2 convolutional layers, feeting into 3 linear layers
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
        x = self.pool(F.relu(self.conv1(x)))  # shape [4, 6, 14, 14]
        x = self.pool(F.relu(self.conv2(x))) # shape [4, 16, 5, 5]
        x = x.view(-1, 16 * 5 * 5) # shape [4, 400]
        x = F.relu(self.fc1(x))  # shape [4, 120]
        x = F.relu(self.fc2(x))  # shape [4, 84]
        x = self.fc3(x) # shape [4, 10]
        return x

def get_filename(plot):
    filename, file_extension = splitext(plot)
    if len(file_extension) == 0:
        filename = f'{filename}.png'
    return filename

def train(root='./data', show=False, lr=0.001, momentum=0.9, n=1000, freq=20, plot='./train.png',
          PATH='./cifar_net.pth', figs='./figs',transform=None):
    '''
    Train network against CIFAR10 data

    Parameters:
        root
        show
        lr
        momentum
        n
        freq
        plot
        PATH
        figs
        transform
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
            inputs, labels = data
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % freq == 1:
                print(f'{epoch + 1}, {i + 1}, {running_loss / freq}')
                losses.append(running_loss/freq)
                running_loss = 0.0

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
        root
        PATH
        transform
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
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

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
    parser.add_argument('--action', choices=['train', 'test'], default='train')
    parser.add_argument('--n', default=10, type=int)
    parser.add_argument('--freq', default=5, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--momentum', default=0.9, type=int)
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--PATH', default='./cifar_net.pth')
    parser.add_argument('--root', default='./data')
    parser.add_argument('--plot', default='./train.png')
    parser.add_argument('--figs', default='./figs', help='Location for storing plot files')
    return parser.parse_args()

def create_transform():
    '''
    '''
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5),
                                                    (0.5, 0.5, 0.5))])

if __name__ == '__main__':
    start  = time()
    args = parse_args()

    match args.action:
        case 'train':
            train(root=args.root, show=args.show, lr=args.lr, momentum=args.momentum, n=args.n,
                  freq=args.freq, plot=args.plot, PATH=args.PATH,figs=args.figs, transform=create_transform())

        case 'test':
            test(root=args.root, PATH=args.PATH, transform=create_transform())

    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')

    if args.show:
        show()
