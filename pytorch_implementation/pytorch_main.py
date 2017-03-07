"""
Main run file for the the CNN in PyTorch
"""

import logging
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import dataset

from pytorch_implementation.pytorch_plant_dataset import PlantDataset

logging.basicConfig(level=logging.DEBUG)


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (2, 1, 0)))
    plt.show()


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


def train(net, train_loader, criterion, optimizer, cuda=False, epochs=5):
    print('Start Training')
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            if cuda:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # update loss
            running_loss += loss.data[0]

        # show status
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss))
        running_loss = 0.0
    print('Finished Training')


def test(net, test_loader, classes):
    # initialize parameters for calculating accuracy
    correct = 0
    total = 0
    class_correct = list(0. for i in range((len(classes))))
    class_total = list(0. for i in range(len(classes)))

    for data in test_loader:
        images, labels = data
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

        # calculate accuracy per class
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1

    # total accuracy
    accuracy = round(100 * correct / float(total), 1)
    print('Accuracy of the network on test images: ', str(accuracy), '%')
    # print('Accuracy of the network on test images: %d %%' % round(100 * correct / float(total), 1))

    # accuracy per class
    for i in range(len(classes)):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))


def run_train_plant(train_set, batch_size=4):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    net = Net()

    # Save network
    saved_net = net.state_dict()
    pickle.dump(saved_net, open('data/trained_network.p', 'wb'))

    criterion = nn.CrossEntropyLoss()  # use a Classification Cross-Entropy loss
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    train(net, train_loader, criterion, optimizer)


def run_test_plant(test_set, classes, batch_size=4):
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)

    saved_net = pickle.load(open('data/trained_network.p', 'wb'))
    test(saved_net, test_loader, classes)


def single_prediction(net, test_loader, classes, batch_size):
    # determine images
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    # show true labels
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

    # show prediction
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j][0]] for j in range(batch_size)))

    # show image
    imshow(torchvision.utils.make_grid(images))


if __name__ == '__main__':
    classes = ['rose', 'sunflower', 'daisy', 'hyacinth',
               'chlorophytum_comosum', 'tradescantia_zebrina', 'philodendron_scandens']

    batch_size = 1

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

    train_set = PlantDataset(root='data/plantset', transform=transform)
    test_set = PlantDataset(root='data/plantset', transform=transform)

    run_train_plant(train_set, batch_size)
    run_test_plant(test_set, classes, batch_size)

    # single_prediction(net, test_loader, classes, batchSize)
