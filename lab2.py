# Lab Session 2

# The objectives of this lab session are the following:
# - Visualize augmented data samples
# - Experiment with Data Augmentation
# - Implement mixup in your training loop

import torchvision.transforms as transforms
import numpy as np
import torch

from matplotlib import pyplot as plt
from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import v2


# Our version of transforms
transform_train = transforms.Compose([
    transforms.Pad(4, padding_mode='reflect'),
    transforms.RandomCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

rootdir = './data/cifar10'

c10train = CIFAR10(rootdir,train=True,download=True,transform=transform_train)
c10test = CIFAR10(rootdir,train=False,download=True,transform=transform_test)

trainloader = DataLoader(c10train,batch_size=9,shuffle=True)
testloader = DataLoader(c10test,batch_size=9,shuffle=True)

f = plt.figure(figsize=(10,10))

# Image with DA for Train batch
for i,(data,target) in enumerate(trainloader):
    
    data = (data.numpy())
    print(data.shape)
    plt.subplot(3,3,1)
    plt.imshow(data[0].swapaxes(0,2).swapaxes(0,1))
    plt.subplot(3,3,2)
    plt.imshow(data[1].swapaxes(0,2).swapaxes(0,1))
    plt.subplot(3,3,3)
    plt.imshow(data[2].swapaxes(0,2).swapaxes(0,1))
    plt.subplot(3,3,4)
    plt.imshow(data[3].swapaxes(0,2).swapaxes(0,1))
    plt.subplot(3,3,5)
    plt.imshow(data[4].swapaxes(0,2).swapaxes(0,1))
    plt.subplot(3,3,6)
    plt.imshow(data[5].swapaxes(0,2).swapaxes(0,1))
    plt.subplot(3,3,7)
    plt.imshow(data[6].swapaxes(0,2).swapaxes(0,1))
    plt.subplot(3,3,8)
    plt.imshow(data[7].swapaxes(0,2).swapaxes(0,1))
    plt.subplot(3,3,9)
    plt.imshow(data[8].swapaxes(0,2).swapaxes(0,1))

    break
f.savefig('images/train_DA.png')

# Image with DA for Test batch
for i,(data,target) in enumerate(trainloader):
    
    data = (data.numpy())
    print(data.shape)
    plt.subplot(3,3,1)
    plt.imshow(data[0].swapaxes(0,2).swapaxes(0,1))
    plt.subplot(3,3,2)
    plt.imshow(data[1].swapaxes(0,2).swapaxes(0,1))
    plt.subplot(3,3,3)
    plt.imshow(data[2].swapaxes(0,2).swapaxes(0,1))
    plt.subplot(3,3,4)
    plt.imshow(data[3].swapaxes(0,2).swapaxes(0,1))
    plt.subplot(3,3,5)
    plt.imshow(data[4].swapaxes(0,2).swapaxes(0,1))
    plt.subplot(3,3,6)
    plt.imshow(data[5].swapaxes(0,2).swapaxes(0,1))
    plt.subplot(3,3,7)
    plt.imshow(data[6].swapaxes(0,2).swapaxes(0,1))
    plt.subplot(3,3,8)
    plt.imshow(data[7].swapaxes(0,2).swapaxes(0,1))
    plt.subplot(3,3,9)
    plt.imshow(data[8].swapaxes(0,2).swapaxes(0,1))

    break
f.savefig('images/test_DA.png')


## MixUP Part 3 Approach
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)