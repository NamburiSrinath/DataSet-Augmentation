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

train_set = datasets.ImageFolder(root='./train_images/', transform=data_transform)    
trainset, traintestset = random_split(train_set, [math.ceil(len(train_set) * 0.85), math.floor(len(train_set) * 0.15)])  

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)                               
traintestsetloader = torch.utils.data.DataLoader(traintestset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)                                      
logging.info('Train set: generated')

testset = datasets.ImageFolder(root='/hdd2/srinath/Imge_net_images', transform=transform)
# testset = torchvision.datasets.CIFAR10(root='CIFAR_10/original/images', train=False,
#                                       download=True, transform=transform)

print("TrainSet, Validation Set, Test Set ", len(trainset), len(traintestset), len(testset))


testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
logging.info('Test set: ImageNet')

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
model.fc = nn.Linear(512, num_classes)

# load the saved model
model.load_state_dict(torch.load("./ImageNet.pth"))
model.eval()

model = model.to('cuda')

print("Test on Generated Test set:")
logging.info("Test on Generated test set:")
TEST(model, traintestsetloader)

print("Test on ImageNet test set:")
logging.info("Test on ImageNet  test set:")
TEST(model, testloader)
