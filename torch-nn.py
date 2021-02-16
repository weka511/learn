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

def train(args,trainloader):
    net       = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    losses    = []

    for epoch in range(args.n):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data # get the inputs; data is a list of [inputs, labels]
            optimizer.zero_grad()  # zero the parameter gradients

            # forward + backward + optimize
            outputs = net(inputs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                losses.append(running_loss/2000)
                running_loss = 0.0

    print('Finished Training')
    torch.save(net.state_dict(), args.PATH)

    fig = plt.figure(figsize=(10,10))
    plt.plot(losses)
    plt.title('Training Loss')

def test(args):

    testset     = torchvision.datasets.CIFAR10(root  = './data',
                                           train     = False,
                                           download  = True,
                                           transform = transform)

    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size  = 4,
                                             shuffle     = False,
                                             num_workers = 0)

    classes    = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    net = Net()
    net.load_state_dict(torch.load(args.PATH))
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

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
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


    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


if __name__=='__main__':

    argparser = argparse.ArgumentParser('Neural Net with Pytorch')
    argparser.add_argument('--train',    default = False, action = 'store_true')
    argparser.add_argument('--test',     default = False, action = 'store_true')
    argparser.add_argument('--n',        default = 2,     type=int)
    argparser.add_argument('--lr',       default = 0.001, type=float)
    argparser.add_argument('--momentum', default = 0.9,   type=int)
    argparser.add_argument('--show',     default = False, action = 'store_true')
    argparser.add_argument('--PATH',     default = './cifar_net.pth')
    args        = argparser.parse_args();

    transform   = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5),
                                                           (0.5, 0.5, 0.5))])

    trainset    = torchvision.datasets.CIFAR10(root      = './data',
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

    if args.train:
        train(args,trainloader)

    if args.test:
        test(args)

    plt.show()
