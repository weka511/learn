# Copyright (C) 202a Greenweaves Software Limited

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

# Torch snippets snarfed from
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def imshow(img):
    img   = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(args):
    trainset    = torchvision.datasets.CIFAR10(root      = args.root,
                                               train     = True,
                                               download  = True,
                                               transform = transform)

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size  = 4,
                                              shuffle     = True,
                                              num_workers = 0)



    if args.show:
        # get some random training images
        dataiter = iter(trainloader)
        images, labels = dataiter.next()

        # show images
        imshow(torchvision.utils.make_grid(images))
        # print labels
        print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    net       = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),
                          lr       = args.lr,
                          momentum = args.momentum)
    losses    = []

    for epoch in range(args.n):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            if i>2: break
            inputs, labels = data # get the inputs; data is a list of [inputs, labels]
            optimizer.zero_grad()  # zero the parameter gradients

            # forward + backward + optimize
            outputs = net(inputs)
            print (outputs)
            print (labels)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if (i +1) % args.freq == 0:
                print(f'{epoch + 1}, {i + 1}, {running_loss / args.freq}')
                losses.append(running_loss/args.freq)
                running_loss = 0.0

    print('Finished Training')
    torch.save(net.state_dict(), args.PATH)

    plt.figure(figsize=(10,10))
    plt.plot(losses)
    plt.ylim(bottom=0)
    plt.title(f'Training Loss: lr={args.lr}, momentum={args.momentum}')
    filename, file_extension = os.path.splitext(args.plot)
    if len(file_extension)==0:
        filename = f'{filename}.png'
    plt.savefig(filename)

def test(root,PATH):

    testset        = torchvision.datasets.CIFAR10(root      = root,
                                                  train     = False,
                                                  download  = True,
                                                  transform = transform)

    testloader     = torch.utils.data.DataLoader(testset,
                                                 batch_size  = 4,
                                                 shuffle     = False,
                                                 num_workers = 0)

    classes        = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    dataiter       = iter(testloader)
    images, labels = dataiter.next()

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    net = Net()
    net.load_state_dict(torch.load(PATH))
    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))

    correct = 0
    total   = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs        = net(images)
            _, predicted   = torch.max(outputs.data, 1)
            total         += labels.size(0)
            correct       += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    class_correct = list(0. for _ in range(len(classes)))
    class_total   = list(0. for _ in range(len(classes)))

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs        = net(images)
            _, predicted   = torch.max(outputs, 1)
            c              = (predicted == labels).squeeze()
            for i in range(4):
                label                 = labels[i]
                class_correct[label] += c[i].item()
                class_total[label]   += 1


    for i in range(len(classes)):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


if __name__=='__main__':

    argparser = argparse.ArgumentParser('Neural Net with Pytorch')
    argparser.add_argument('--train',    default = False, action = 'store_true')
    argparser.add_argument('--test',     default = False, action = 'store_true')
    argparser.add_argument('--n',        default = 2,     type=int)
    argparser.add_argument('--freq',     default = 2000,  type=int)
    argparser.add_argument('--lr',       default = 0.001, type=float)
    argparser.add_argument('--momentum', default = 0.9,   type=int)
    argparser.add_argument('--show',     default = False, action = 'store_true')
    argparser.add_argument('--PATH',     default = './cifar_net.pth')
    argparser.add_argument('--root',     default = './data')
    argparser.add_argument('--plot',     default = './train.png')
    args        = argparser.parse_args();

    transform   = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5),
                                                           (0.5, 0.5, 0.5))])


    if args.train:
        train(args)

    if args.test:
        test(args.root,args.PATH)

    plt.show()
