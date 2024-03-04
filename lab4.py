import numpy as np 
import torchvision.transforms as transforms
import torch 
import torchvision
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune

import os
import argparse
import time

from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR10
from torch.autograd import Variable
from torchsummary import summary

from binnaryconnect import BC

## Argument parser
parser = argparse.ArgumentParser(description='Lab EFFDL')
parser.add_argument('--model', default='PreActResNet18', type=str, help='Options: ResNet18, PreActResNet18, DenseNet121, VGG19')
parser.add_argument('--lr', default=0.05, type=float, help='Learning rate')
parser.add_argument('--epochs', default=5, type=int, help='Number of epochs')
parser.add_argument('--alpha', default=1.0, type=float, help='Alpha value for Mixup')
parser.add_argument('--amount', default=0.3, type=float, help='Alpha value for Mixup')
args = parser.parse_args()

## Test if there is GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device , "\n")

## Data loading
## Normalization adapted for CIFAR10
normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

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

## Example of usage of the models
from model_cifar100 import *

## We can use the function contar_parametros to count the number of parameters in the model
def contar_parametros(modelo):
    return sum(p.numel() for p in modelo.parameters())

print("\n== Builind the model ==")

net = PreActResNet18()
net = net.to(device)

# Flatten all weights into a single vector
all_weights = torch.cat([p.flatten() for p in net.parameters()])

# Compute the threshold
threshold = torch.quantile(all_weights.abs(), args.amount)
print(threshold)

# Apply global pruning
for name, module in net.named_modules():
    if isinstance(module, nn.Conv2d):
        # Use a custom pruning function that prunes weights below the threshold
        prune.custom_from_mask(module, name="weight", mask=module.weight.abs() > threshold)
        prune.remove(module, name="weight")
    if isinstance(module, nn.Conv2d):
        # Use a custom pruning function that prunes weights below the threshold
        prune.custom_from_mask(module, name="weight", mask=module.weight.abs() > threshold)
        prune.remove(module, name="weight")


# Iterate through all the convolutional layers
# for name, module in net.named_modules():
#     if isinstance(module, nn.Conv2d):
#         prune.l1_unstructured(module, name="weight", amount=args.amount)
#         prune.remove(module, name="weight")

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

## Training the model
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

start = time.time()

for i in range(max_epochs):
    train(i)
    test(i)
    scheduler.step()

print(f"\nBest accuracy: {best_acc}")
# print(f"Number of parameter AFTER Prune: {contar_parametros(net)}")

end = time.time()
# print(f"\nTime to train the model: {(end-start)/60} minutes")

with open('results_presentation.txt', 'a') as f:
    f.write(f'\n{args.model}; {max_epochs}; {lr}; {best_acc}; {contar_parametros(net)}; {(end-start)/60}; {args.amount}; {train_losses}; {test_losses}' )

# ## Plotting the results
# import matplotlib.pyplot as plt
# epochs = range(1, max_epochs+1)

# plt.plot(epochs, train_losses, label='Training Loss')
# plt.plot(epochs, test_losses, label='Test Loss')
# plt.title('Loss Plotting')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()

# save_path = f'images/loss_plot_{max_epochs}_{lr}_{args.mixup}_{args.dataaug}.png'
# plt.savefig(save_path)