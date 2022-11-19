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
import matplotlib.pyplot as plt
import math
import random
import logging
import os
from datetime import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
g = torch.Generator()
g.manual_seed(0)

# writer = SummaryWriter("../tensorboard_logs")

if not os.path.exists("logs"):
    os.mkdir("logs")
logging.basicConfig(filename='logs/experiment_50epochs.log', format='%(asctime)s %(message)s', level=logging.INFO)

train_val_ratio = 0.85
no_of_epochs = 50
batch_size = 512
num_classes = 10
pretrained = False 
tunable = True
logging.info(f'No of epochs: {no_of_epochs}, pretrained: {pretrained}, tunable: {tunable}')

classes = ['airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

label_list = ['airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# train_folder = '/hdd2/srinath/dataset_augmentation_diffusers/train_images/'
train_folder = '/hdd2/srinath/Dreambooth-Stable-Diffusion/generated_images'
# test_folder = '/hdd2/srinath/Imge_net_images'
test_folder = '/hdd2/srinath/dataset_augmentation_diffusers/imagenet_val/'

# ImageNet transformer
transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

# Generated data transformer
training_data_transform = transforms.Compose([
        transforms.RandAugment(magnitude=15),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
# class AverageMeter(object):
#     """Computes and stores the average and current value"""

#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.val = 0.0
#         self.avg = 0.0
#         self.sum = 0
#         self.count = 0.0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count

def train_validation_test_splits(train_folder, test_folder, train_val_ratio, train_transform = transform):
    '''
    Given train and test folders, split the dataset into train, validation and test sets
    Train and Validation data is splitted in the train_val_ratio

    Returns:
        Train, Validation and test data loader
    '''
    train_set = datasets.ImageFolder(root=train_folder, transform=train_transform)
    trainset_size = math.ceil(len(train_set) * train_val_ratio)
    valset_size = len(train_set) - trainset_size
    trainset, validationset = random_split(train_set, [trainset_size, valset_size])  

    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    validation_dataloader = torch.utils.data.DataLoader(validationset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    logging.info('Train set: generated')
    testset = datasets.ImageFolder(root=test_folder, transform=transform)
    print("TrainSet, Validation Set, Test Set ", len(trainset), len(validationset), len(testset))
    print("Training data and test data ", train_folder, test_folder)
    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    logging.info('Test set: ImageNet')
    return train_dataloader, validation_dataloader, test_dataloader

def set_parameter_requires_grad(model, tunable):
	'''
    	We would either want to 
    	    1. Freeze the backbone i.e tunable = False -> requires_grad = False
    	    2. Tunable backbone i.e tunable = True -> requires_grad = True
    	
    	Returns: 
    	    The model with either frozen or tunable backbone with a Linear head
    	'''
	for param in model.parameters():
		param.requires_grad = tunable
	model.fc = nn.Linear(512, num_classes)
	return model

def verify_freeze(model):
    '''
    Verify whether the backbone is frozen or tunable

    Returns:
        1. params_to_update -> Parameters to update which is used by Optimizer
        2. Layers to update -> Names of layers for which backpropagation occurs (just to verify)
    '''
    params_to_update = []
    layers_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            layers_update.append(name)
    return params_to_update, layers_update

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def train(net, train_loader, no_of_epochs, criterion, optimizer):

    # batch_time = AverageMeter()
    # data_time = AverageMeter()
    # losses = AverageMeter()
    # top1 = AverageMeter()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    num_batches = len(train_loader)

    for epoch in range(no_of_epochs):  # loop over the dataset multiple times
        correct = 0
        total = 0
        running_loss = 0.0
        
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            # inputs, labels = data
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # acc1, acc5 = accuracy(output, target, topk=(1, 5))
            # losses.update(loss.item(), input.size(0))
            # top1.update(acc1.item(), input.size(0))
            # top5.update(acc5.item(), input.size(0))

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            
            if i == (num_batches - 1):    # print at the end of batch
                print(f'[Epoch: {epoch + 1}, Batch: {i + 1:5d}] loss: {running_loss / num_batches:.3f}')
                logging.info(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / num_batches:.3f}')
                running_loss = 0.0
                _, predicted = torch.max(outputs.data.cuda(), 1)
                total += labels.size(0)
                correct += (predicted == labels.cuda()).sum().item()
                print(f'Training Accuracy: {100 * correct // total} %')
                logging.info(f'Training Accuracy: {100 * correct // total} %')

        #         writer.add_scalar("data/training_loss", losses.val, i)
        # writer.add_scalars("data/top1_accuracy", {"train": top1.avg}, epoch + 1)
        # writer.add_scalars("data/top5_accuracy", {"train": top5.avg}, epoch + 1)

    print('Finished Training')
    logging.info('Finished Training')

    # Commented now as we don't need to save .pth files
    # now = datetime.now()
    # dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    # PATH = './ImageNet_'+ str(len(train_set)) + dt_string + '.pth'
    # torch.save(net.state_dict(), PATH)

def test(net, test_loader):
    correct = 0
    total = 0
    net.eval()
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

if __name__ == "__main__":
    # Get dataloaders
    train_dataloader, validation_dataloader, test_dataloader = train_validation_test_splits(train_folder, test_folder, 
                                                                                train_val_ratio, training_data_transform)

    # Get model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=pretrained)              
    model = set_parameter_requires_grad(model, tunable)
    model = model.to('cuda')
    params_to_update, layers_update = verify_freeze(model)
    print("No of layers backprop is going is :", len(layers_update))

    # Initialize optimizer and loss function
    optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Train
    train(model, train_dataloader, no_of_epochs, criterion, optimizer)

    print("------ Validation starts-----", train_folder)
    logging.info(f"Validation on {train_folder} test set:")
    test(model, validation_dataloader)

    print("------ Test starts-----", test_folder)
    logging.info(f"Test on {test_folder} test set:")
    test(model, test_dataloader)
