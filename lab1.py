import numpy as np 
import torchvision.transforms as transforms
import torch 
import torchvision
import torch.optim as optim

import os
import argparse

from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR10

parser = argparse.ArgumentParser(description='Lab01 Training/Testing Model')
parser.add_argument('--model', default='ResNet18', type=str, help='Options: ResNet18, PreActResNet18, DenseNet121, VGG19')
parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
parser.add_argument('--epochs', default=5, type=int, help='Number of epochs')

args = parser.parse_args()

## Test if there is GPU
device = torch.device("cuda" if torch.cuda.is_available() else "CPU")
print(device)

## Normalization adapted for CIFAR10
normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

# Transforms is a list of transformations applied on the 'raw' dataset before the data is fed to the network. 
# Here, Data augmentation (RandomCrop and Horizontal Flip) are applied to each batch, differently at each epoch, on the training set data only
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
    model = models.get('ResNet18')

net = model
net = net.to(device)

## Hyperparameters to set
lr = args.lr
criterion = nn.CrossEntropyLoss()
max_epochs = args.epochs
best_acc = 0
optimizer = optim.SGD(net.parameters(), lr=lr)

## Save results
results = []
params = []

## Training the model
def train(epoch):
    
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
    # for batch_idx, (inputs, targets) in enumerate(trainloader_subset):
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

    # print(f"Train Loss: {train_loss/(batch_idx+1)}")

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


for i in range(max_epochs):
    train(i)
    test(i)

print(f"Best accuracy: {best_acc}")
results.append(best_acc)
print(f"Number of parameters in the model: {contar_parametros(net)}")
params.append(contar_parametros(net))


# ## Task 2 Just plotting the table
# import matplotlib.pyplot as plt

# Acc = [93.02, 95.11, 95.04, 92.64]
# Params = [11220132, 11217316, 7048548, 20086692]

# fig, ax = plt.subplots()
# ax.scatter(Params, Acc)

# ax.annotate('ResNet18', (Params[0], Acc[0]))
# ax.annotate('PreActResNet18', (Params[1], Acc[1]))
# ax.annotate('DenseNet121', (Params[2], Acc[2]))
# ax.annotate('VGG19', (Params[3], Acc[3]))

# plt.xlabel('Number of parameters ')
# plt.ylabel('Accuracy (%)')
# plt.title('Number of parameters vs Accuracy')
# plt.savefig('images/params_vs_acc.png')
