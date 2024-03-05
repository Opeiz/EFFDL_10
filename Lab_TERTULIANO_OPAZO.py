import numpy as np 
import torchvision.transforms as transforms
import torch 
import torchvision
import torch.optim as optim

import os
import argparse
import time

from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR10
from torch.autograd import Variable

## Argument parser
parser = argparse.ArgumentParser(description='Lab EFFDL')
parser.add_argument('--model', default='PreActResNet18', type=str, help='Options: ResNet18, PreActResNet18, DenseNet121, VGG19')
parser.add_argument('--lr', default=0.05, type=float, help='Learning rate')
parser.add_argument('--epochs', default=5, type=int, help='Number of epochs')
parser.add_argument('--alpha', default=1.0, type=float, help='Alpha value for Mixup')
parser.add_argument('--dataaug', action='store_true', help='Use data augmentation or not')
parser.add_argument('--mixup', action='store_true', help='Use Mixup or not')
args = parser.parse_args()

## Test if there is GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

## Data loading
## Normalization adapted for CIFAR10
normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

# Transforms is a list of transformations applied on the 'raw' dataset before the data is fed to the network. 
# Here, Data augmentation (RandomCrop and Horizontal Flip) are applied to each batch, differently at each epoch, on the training set data only
if args.dataaug:
    transform_train = transforms.Compose([
        transforms.Pad(4, padding_mode='reflect'),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        normalize_scratch,
    ])
else:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize_scratch,
    ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize_scratch,
])

### The data from CIFAR10 will be downloaded in the following folder
rootdir = './data/cifar10'

c10train = CIFAR10(rootdir,train=True,download=True,transform=transform_train)
c10test = CIFAR10(rootdir,train=False,download=True,transform=transform_test)

trainloader = DataLoader(c10train,batch_size=32,shuffle=True)
testloader = DataLoader(c10test,batch_size=32)


## number of target samples for the final dataset
num_train_examples = len(c10train)
num_samples_subset = 15000

## We set a seed manually so as to reproduce the results easily
seed  = 2147483647

## Generate a list of shuffled indices ; with the fixed seed, the permutation will always be the same, for reproducibility
indices = list(range(num_train_examples))
np.random.RandomState(seed=seed).shuffle(indices)## modifies the list in place

## We define the Subset using the generated indices 
c10train_subset = torch.utils.data.Subset(c10train,indices[:num_samples_subset])
print(f"Initial CIFAR10 dataset has {len(c10train)} samples")
print(f"Subset of CIFAR10 dataset has {len(c10train_subset)} samples")

# Finally we can define anoter dataloader for the training data
trainloader_subset = DataLoader(c10train_subset,batch_size=32,shuffle=True)

## Model definition
## Choosing the model
## - ResNet
## - PreActResNet
## - DenseNet
## - VGG

## Example of usage of the models
from model_cifar100 import *

print("== Builind the model ==")
models = {'ResNet18': ResNet18(), 'PreActResNet18': PreActResNet18(), 'DenseNet121': DenseNet121(), 'VGG19': VGG('VGG19')}

if args.model in models:
    print(f"\nModel found in the list, going to train :)")
    print(f"Training {args.model} model")
    model = models.get(args.model)
else:
    model = models.get('PreActResNet18')

net = model
net = net.to(device)

## Hyperparameters to set
lr = args.lr
criterion = nn.CrossEntropyLoss()
max_epochs = args.epochs
best_acc = 0
optimizer = optim.SGD(net.parameters(), lr=lr)
alpha = args.alpha
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

## Save results
results = []
params = []
train_losses = []
test_losses = []

## MixUp
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

## Training the model
def trainMX(epoch):
    
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha, True)
        inputs, targets_a, targets_b = map(Variable, (inputs,targets_a, targets_b))

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_losses.append(train_loss / len(trainloader))
    print(f"Train Loss: {train_loss/(batch_idx+1)}")

def train(epoch):
    
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_losses.append(train_loss / len(trainloader))
    print(f"Train Loss: {train_loss/(batch_idx+1)}")

# Testing the model
def test(epoch):
    
    global best_acc
    net.eval()

    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_losses.append(test_loss / len(testloader))
    print(f"Test Loss: {test_loss/(batch_idx+1)}")

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        pathckpt = f'./ckpts/ckpt_{args.model}.pth'
        torch.save(state, pathckpt)
        best_acc = acc

# Function to count the parameters in the model
def contar_parametros(modelo):
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)

start = time.time()

for i in range(max_epochs):
    if args.mixup:
        trainMX(i)
    else:
        train(i)
    test(i)
    scheduler.step()

print(f"Best accuracy: {best_acc}")
print(f"Number of parameters in the model: {contar_parametros(net)}")

end = time.time()
print(f"\nTime to train the model: {(end-start)/60} minutes")

with open('results.txt', 'a') as f:
    f.write(f'\n{args.model}, {max_epochs}, {lr}, {best_acc}, {contar_parametros(net)}, {(end-start)/60}, {args.dataaug}, {args.mixup}' )

## Plotting the results
import matplotlib.pyplot as plt

epochs = range(1, max_epochs+1)

plt.plot(epochs, train_losses, label='Training Loss')
plt.plot(epochs, test_losses, label='Test Loss')
plt.title('Loss Plotting')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

save_path = f'images/loss_plot_{max_epochs}_{lr}_{args.mixup}_{args.dataaug}.png'

plt.savefig(save_path)