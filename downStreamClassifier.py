import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
# from pylab import *
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import math

import random
import logging
import os
from datetime import datetime
import numpy as np

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
g = torch.Generator()
g.manual_seed(0)

if not os.path.exists("logs"):
    os.mkdir("logs")
logging.basicConfig(filename='logs/prompting_log.log', format='%(asctime)s %(message)s', level=logging.INFO)

# ImageNet transformer
transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# Generated data transformer
data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

batch_size = 32
num_classes = 10

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

label_list = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# train_set = datasets.ImageFolder(root='./train_images/', transform=data_transform)    
train_set = datasets.ImageFolder(root='../Dreambooth-Stable-Diffusion/generated_images/', transform=data_transform)    
trainset_size = math.ceil(len(train_set) * 0.85)
trainset, traintestset = random_split(train_set, [math.ceil(len(train_set) * 0.85), math.floor(len(train_set) * 0.15)])  

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2, generator=g)
traintestsetloader = torch.utils.data.DataLoader(traintestset, batch_size=batch_size,
                                          shuffle=True, num_workers=2, generator=g)

logging.info('Train set: generated')

testset = datasets.ImageFolder(root='/hdd2/srinath/Imge_net_images', transform=transform)
# testset = torchvision.datasets.CIFAR10(root='CIFAR_10/original/images', train=False,
#                                       download=True, transform=transform)

print("TrainSet, Validation Set, Test Set ", len(trainset), len(traintestset), len(testset))


testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
logging.info('Test set: ImageNet')


def train(net, train_loader):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    print(net.parameters())
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    num_batches = trainset_size // batch_size

    for epoch in range(10):  # loop over the dataset multiple times

        correct = 0
        total = 0
        running_loss = 0.0
        
        for i, data in enumerate(train_loader, 0):

            # get the inputs; data is a list of [inputs, labels]
            # inputs, labels = data
            inputs, labels = data[0].to(device), data[1].to(device)
            
            # print label
            # print(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            
            if i == (num_batches-1):    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 500:.3f}')
                logging.info(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 500:.3f}')
                running_loss = 0.0
                _, predicted = torch.max(outputs.data.cuda(), 1)
                total += labels.size(0)
                correct += (predicted == labels.cuda()).sum().item()
                print(f'Accuracy: {100 * correct // total} %')
                logging.info(f'Accuracy: {100 * correct // total} %')

    print('Finished Training')
    logging.info('Finished Training')

    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    PATH = './ImageNet_'+ str(len(train_set)) + dt_string + '.pth'
    torch.save(net.state_dict(), PATH)

def TEST(net, test_loader):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images.cuda())
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data.cuda(), 1)
            total += labels.size(0)
            correct += (predicted == labels.cuda()).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct // total} %')
    logging.info(f'Accuracy of the network on the test images: {100 * correct // total} %')

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images.cuda())
            _, predictions = torch.max(outputs.cuda(), 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label.cuda() == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
        logging.info(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

feature_extract = True

# We will just finetune (so won't pass gradient to back)
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
                   
set_parameter_requires_grad(model, feature_extract)
model.fc = nn.Linear(512, num_classes)

model = model.to('cuda')

params_to_update = model.parameters()

print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

train(model, trainloader)

print("Test on Generated Test set:")
logging.info("Test on Generated test set:")
TEST(model, traintestsetloader)

print("Test on ImageNet test set:")
logging.info("Test on ImageNet test set:")
TEST(model, testloader)
