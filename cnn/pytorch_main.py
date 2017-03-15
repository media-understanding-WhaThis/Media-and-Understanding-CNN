"""
Main run file for the the CNN in PyTorch
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from cnn.image_processor import ImageProcessor
from cnn.pytorch_plant_dataset import PlantDataset
from torch.autograd import Variable
from torch.utils.data import dataset

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
        self.pool = nn.MaxPool2d(2, 2)
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


def train(net, train_loader, criterion, optimizer, test_set, batch_size, cuda=False, epochs=30):
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

        if epoch % 5 == 0:
            test_loader = torch.utils.data.DataLoader(
                test_set, batch_size=batch_size, shuffle=True, num_workers=2)
            test(net, test_loader, classes)
    print('Finished Training')


def test(net, test_loader, classes):
    # initialize parameters for calculating accuracy
    correct = 0
    total = 0
    class_correct = list(0. for i in range((len(classes))))
    class_total = list(0. for i in range(len(classes)))

    for data in test_loader:
        images, labels = data
        # Variable(images) = [torch.FloatTensor of size 4x3x32x32]
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

    # accuracy per class
    for i in range(len(classes)):
        accuracy = round(100 * class_correct[i] / float(class_total[i]), 1)
        print('Accuracy of ', classes[i], ': ', str(accuracy), '%')


def run_train_plant(train_set, test_set, batch_size=4):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    net = Net()

    criterion = nn.CrossEntropyLoss()  # use a Classification Cross-Entropy loss
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    train(net, train_loader, criterion, optimizer, test_set, batch_size)

    # Save network
    saved_net = net.state_dict()
    torch.save(net, 'data/trained_network.p')


def run_test_plant(test_set, classes, batch_size=4):
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)
    saved_net = torch.load('data/trained_network.p')
    test(saved_net, test_loader, classes)


def single_prediction(image_path, classes):
    saved_net = torch.load('data/trained_network.p')

    processor = ImageProcessor(image_path)
    processor.center_crop()
    processor.resize_image()
    processed_image = np.asarray(processor.im, dtype=np.float32)

    processed_tensor = torch.from_numpy(np.transpose(processed_image))
    squeezed = processed_tensor.unsqueeze(0)

    outputs = saved_net(Variable(squeezed))
    _, predicted = torch.max(outputs.data, 1)
    predicted_num = predicted[0][0]

    return predicted_num, classes[predicted_num]


def used_classes():
    return ['rose', 'sunflower', 'daisy', 'forget-me-not']


if __name__ == '__main__':
    batch_size = 4

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

    train_set = PlantDataset(root='data/plantset', transform=transform, train=True)
    test_set = PlantDataset(root='data/plantset', transform=transform, train=False)

    run_train_plant(train_set, test_set, batch_size)
    run_test_plant(test_set, used_classes(), batch_size)

    # saved_net = torch.load('data/trained_network.p')
    # predicted_num, classes[predicted_num] = single_prediction(
    #    saved_net, 'data/plantset/sunflower/1001901836_3d592b5f93.jpg', classes)
    # print(classes[predicted_num])
