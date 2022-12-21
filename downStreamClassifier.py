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
from torch.utils.data import Subset
from collections import defaultdict

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
g = torch.Generator()
g.manual_seed(0)

# writer = SummaryWriter("../tensorboard_logs")

no_of_epochs = 300
final_dataset_length = 10000
train_ratio = 0.50
test_ratio = 0.20
val_ratio = 1 - train_ratio - test_ratio
print(f'{train_ratio = }, {val_ratio = }, {test_ratio = }')
batch_size = 512
num_classes = 10
proportion = 0
pretrained = False
tunable = True
if not os.path.exists("logs"):
    os.mkdir("logs")
logging.basicConfig(filename=f'logs/tests/test_custom/experiment_{no_of_epochs}epochs.log', format='%(asctime)s %(message)s', level=logging.INFO)

logging.info("----------------------Experiment starts-----------------------------------------")
logging.info(f'No of epochs: {no_of_epochs}, pretrained: {pretrained}, tunable: {tunable}, proportion: {proportion}')
exp_state = f"e{no_of_epochs}_p{pretrained}_t{tunable}"

classes = ['airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

label_list = ['airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

train_folder = '/hdd2/srinath/srinath/dataset_augmentation_diffusers/custom_test_set/'
augment_folder = '/hdd2/srinath/dataset_augmentation_diffusers/train_images/'
# augment_folder = '/hdd2/srinath/Dreambooth-Stable-Diffusion/generated_images'
test_folder = '/hdd2/srinath/dataset_augmentation_diffusers/custom_test_set/'

logging.info(f'Train: {train_folder}, Augment: {augment_folder}, Test: {test_folder}')

# train_folder = '/hdd2/srinath/Dreambooth-Stable-Diffusion/generated_images'
# test_folder = '/hdd2/srinath/dataset_augmentation_diffusers/custom_test_set/'

# train_folder = '/hdd2/srinath/dataset_augmentation_diffusers/train_images/'
# test_folder = '/hdd2/srinath/dataset_augmentation_diffusers/custom_test_set/'

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

def train_validation_test_splits(train_folder, test_folder, train_val_ratio, train_transform = transform):
    '''
    Given train and test folders, split the dataset into train, validation and test sets
    Train and Validation data is splitted in the train_val_ratio

    Returns:
        Train, Validation and test data loader
    '''
    train_set = train_folder
    # train_set = datasets.ImageFolder(root=train_folder, transform=train_transform)
    trainset_size = math.ceil(len(train_set) * train_val_ratio)
    valset_size = len(train_set) - trainset_size
    trainset, validationset, testset = random_split(train_set, [trainset_size, valset_size, testset_size])  

    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    validation_dataloader = torch.utils.data.DataLoader(validationset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    # logging.info('Train set: generated')
    testset = datasets.ImageFolder(root=test_folder, transform=transform)
    print("TrainSet, Validation Set, Test Set ", len(trainset), len(validationset), len(testset))
    print("Training data and test data ", train_folder, test_folder)
    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    # logging.info('Test set: ImageNet')
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

        print("------ Validation starts-----", train_folder)
        logging.info(f"Validation on {train_folder} test set:")
        test(model, validation_dataloader)


    print('Finished Training')
    logging.info('Finished Training')

    # Commented now as we don't need to save .pth files
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    PATH = './logs/test_custom/ImageNet_'+  exp_state + dt_string + '.pth'
    torch.save(net.state_dict(), PATH)

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

def dataset_mixing(train_folder, augment_folder, final_dataset_length, proportion):
    '''
    Takes in the train_set and augment set and mixes in proportion such that
    Eg: If final dataset length is 10k and proportion is 0.1
        Individual class length - 1k
        Images coming from Original dataset - 1k * (1-0.1) = 900 images
        Images coming from Augmented dataset - 1k * 0.1 = 100 images

    Note: proportion = 0 --> Entire Data is from original dataset
          proportion = 1 ---> Entire Data is from augmented dataset
    
    Note: If original dataset has 100 images for a particular class but it was asked to get 900 images,
          then these 100 images will be repeated 9 times (Refer random.choices documentation)

    This function works only if both train set and augment set labels are same. Else it's working is unexpected
    '''
    train_set = datasets.ImageFolder(root=train_folder, transform=training_data_transform)
    augment_set = datasets.ImageFolder(root=augment_folder, transform=training_data_transform)

    # Get unique class labels
    class_labels = set(train_set.targets)

    # No of images we need to take for each class (it should be same to avoid class imbalance)
    individual_class_length = final_dataset_length/len(class_labels)

    # No of images from original dataset and augmented dataset
    original_length, augmented_length = int((1-proportion)*individual_class_length), int(proportion*individual_class_length)
    train_idx_final = []
    augment_idx_final = []
    for i in class_labels:
        # Get the indices where the target labels are same as the class index and extend it to final list 
        train_idx = np.where(np.array(train_set.targets)==i)[0]
        train_idx_final.extend(random.choices(train_idx, k=original_length))

        augment_idx = np.where(np.array(augment_set.targets)==i)[0]
        augment_idx_final.extend(random.choices(augment_idx, k=augmented_length))
        
        # Uncomment to print and verify
        # print("-----------Train Idx starts---------")
        # print(train_idx)
        # print("-----------Train Idx Final starts---------------------")
        # print(train_idx_final)
        # print("------- Augment Idx starts------------")
        # print(augment_idx)
        # print("-----------Augment Idx Final starts--------------------")
        # print(augment_idx_final)
        # print("-------------------")

    # Lengths should match
    print(f"Actual No of images from training data and augmented dataset are {len(train_idx_final)}, {len(augment_idx_final)}")
    print(f"Expected No of images from training data and augmented dataset are {original_length*len(class_labels)}, {augmented_length*len(class_labels)}")
    
    logging.info(f"Actual No of images from training data and augmented dataset are {len(train_idx_final)}, {len(augment_idx_final)}")
    logging.info(f"Expected No of images from training data and augmented dataset are {original_length*len(class_labels)}, {augmented_length*len(class_labels)}")
    # Train and Augment subsets are nothing but taking these specific datapoints based on indices and concatenating them
    train_subset = Subset(train_set, train_idx_final)
    augment_subset = Subset(augment_set, augment_idx_final)
    final_set = torch.utils.data.ConcatDataset([train_subset, augment_subset])
    return final_set

def verify_dataset_mixing(final_set):
    '''
    Prints no of images present in each class. Is inefficient and takes some time to execute 
    as targets is missng need to go through each datapoint
    '''
    class_dict = defaultdict(int)
    for _, label in final_set:
        class_dict[label] += 1
    print(class_dict)

if __name__ == "__main__":

    final_set = dataset_mixing(train_folder, augment_folder, final_dataset_length, proportion=proportion)
    
    # Optional function, uncomment to see the no of images for each class, will take sometime to execute
    # verify_dataset_mixing(final_set)

    # Note: Observe the change in first argument, we are passing final_set, not the train folder 
    # as now we are mixing from different data folders. 
    train_dataloader, validation_dataloader, test_dataloader = train_validation_test_splits(final_set, test_folder, train_ratio, val_ratio, test_ratio, training_data_transform)

    # Uncomment the below code for previous version

    # Get dataloaders
    # train_dataloader, validation_dataloader, test_dataloader = train_validation_test_splits(train_folder, test_folder, 
    #                                                                           train_val_ratio, training_data_transform)

    # Get model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=pretrained)              
    model = set_parameter_requires_grad(model, tunable)
    model = model.to('cuda')
    params_to_update, layers_update = verify_freeze(model)
    print("No of layers backprop is going is :", len(layers_update))

    # Initialize optimizer and loss function
    # optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    optimizer = optim.adam(params_to_update, lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Train
    train(model, train_dataloader, no_of_epochs, criterion, optimizer)

    print("------ Test starts-----", test_folder)
    logging.info(f"Test on {test_folder} test set:")
    test(model, test_dataloader)
