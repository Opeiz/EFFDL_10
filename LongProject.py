import numpy as np 
import torchvision.transforms as transforms
import torch 
import torchvision
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torchinfo

import os
import argparse
import time

from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR10
from torch.autograd import Variable

from binnaryconnect import BC

## Argument parser
parser = argparse.ArgumentParser(description='Project EFFDL')

# Classic arguments for the model
parser.add_argument('--lr', default=0.05, type=float, help='Learning rate')
parser.add_argument('--epochs', default=5, type=int, help='Number of epochs')
parser.add_argument('--epochs_fine', default=3, type=int, help='Number of epochs for fine tuning')
parser.add_argument('--batch', default=32, type=int, help='Batch size')

# Data Augmentation
parser.add_argument('--da', action='store_true', help='Use data augmentation or not')

# Pruning arguments
parser.add_argument('--prune', action='store_true', help='Prune the model or not')
parser.add_argument('--prune_ratio', default=30 , type=float, help='Amount for Pruning')
parser.add_argument('--fine_tuning', action='store_true', help='Use fine tuning or not')
parser.add_argument('--custom_prune', action='store_true', help='Use custom pruning or not')
parser.add_argument('--structured', action='store_true', help='Use structured pruning or not')

# Quantize arguments
parser.add_argument('--half', action='store_true', help='Use half precision or not')

# Model arguments
parser.add_argument('--factorized', action='store_true', help='Model to use')
parser.add_argument('--factor', default=1, type=int, help='Factor for the factorized model')

# Checkpoint arguments
parser.add_argument("--ckpt", default=None, type=str, help="Path to the checkpoint")

args = parser.parse_args()

# =================================================================================================
## Test if there is GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using : ", device , "\n")

## Data loading
## Normalization adapted for CIFAR10
normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

if args.da:

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        # transforms.RandomRotation(60),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2),
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

batch_size = args.batch
trainloader = DataLoader(c10train,batch_size=batch_size,shuffle=True)
testloader = DataLoader(c10test,batch_size=batch_size)


## number of target samples for the final dataset
num_train_examples = len(c10train)
num_samples_subset = 15000

## We set a seed manually so as to reproduce the results easily
seed  = 2147483647

## Generate a list of shuffled indices ; with the fixed seed, the permutation will always be the same, for reproducibility
indices = list(range(num_train_examples))
np.random.RandomState(seed=seed).shuffle(indices)## modifies the list in place

## We define the Subset using the generated indices 
print(f"Initial CIFAR10 dataset has {len(c10train)} samples")

## Example of usage of the models
from model_cifar100 import *
from model_cifar100.preact_resnet_factorized import PreActResNet18_fact

## We can use the function contar_parametros to count the number of parameters in the model
def sizeModel(modelo):
    return torchinfo.summary(modelo, verbose=0).total_params

print(f"\n== Builind the model ==")

if args.factorized:
    net = PreActResNet18_fact(args.factor)
else:
    net = PreActResNet18()
net = net.to(device)

## Hyperparameters to set
lr = args.lr
criterion = nn.CrossEntropyLoss()
max_epochs = args.epochs
best_acc = 0
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

if args.half:
    net = net.half()
    criterion = criterion.half()

prune_type = "None"

## Save results
results = []
params = []
train_losses = []
test_losses = []
accuracies = []

## Training the model
def train(epoch):
    
    print('\n==> Epoch: %d' % epoch, '\t Best Accuracy: %.3f' % best_acc)

    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        
        inputs, targets = inputs.to(device), targets.to(device)

        if args.half:
            inputs = inputs.half()

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_losses.append(train_loss / len(trainloader))
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

            if args.half:
                inputs = inputs.half()

            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_losses.append(test_loss / len(testloader))
    # print(f"Test Loss: {test_loss/(batch_idx+1)}")

    # Save checkpoint.
    acc = 100.*correct/total
    accuracies.append(acc)

    if acc > best_acc :

        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'lr': lr,
            'optimizer': optimizer,
            'scheduler': scheduler
        }
        if args.factorized:
            pathckpt = f'./ckpts/project/ckpt_fact_Epochs={args.epochs}_Da={args.da}_Half={args.half}.pth'
        else:
            pathckpt = f'./ckpts/project/ckpt_Epochs={args.epochs}_Da={args.da}_Half={args.half}.pth'
        torch.save(state, pathckpt)

        best_acc = acc
    
    return acc

start = time.time()

if args.ckpt:
    print("==> Loading checkpoint")
    assert os.path.isdir('ckpts'), 'Error: no checkpoint directory found!'

    checkpoint = torch.load(args.ckpt)
    net.load_state_dict(checkpoint['net'])
    
    best_acc = checkpoint['acc']
    optimizer = checkpoint['optimizer']
    scheduler = checkpoint['scheduler']
    lr = checkpoint['lr']

else:
    for epoch in range(max_epochs):
        train(epoch)
        test(epoch)
        scheduler.step()

pruned_size = 0
prune_type = None
if args.prune:

    # Modules to prune
    to_prune = []
    for name, module in net.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            to_prune.append(module)

    if args.custom_prune:
        prune_type = "custom"
        print(f"\n ==> Custom Pruning")

        all_weights = torch.cat([p.flatten().float() for p in net.parameters()])
        threshold = torch.quantile(all_weights.abs(), args.prune_ratio/100)

        mask = module.weight.abs() > threshold
        for module in to_prune:
            prune.custom_from_mask(module, name="weight", mask=mask)

    if args.structured:
        prune_type = "structured"
        print(f"\n ==> Structured Pruning")
        for module in to_prune:
            prune.ln_unstructured(module, name="weight", amount=args.prune_ratio/100, dim=1)
   
    else:
        prune_type = "unstructured"
        print(f"\n ==> Unstructured Pruning")
        for module in to_prune:
            prune.l1_unstructured(module, name="weight", amount=args.prune_ratio/100)

    # Fine tuning
    if args.fine_tuning:
        print(f"\n ==> Fine tuning")
        for epoch in range(args.epochs_fine):
            train(epoch)

    pruned_size = sizeModel(net)
    print(f"Number of parameter in PRUNE: {pruned_size}")
    print(f"Pruned by :", args.prune_ratio/100, "%")

    acc = test(0)

    print(f"Accuracy after Pruning :",  acc )

    for name, module in net.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.remove(module, name="weight")

end = time.time()

print(f"\nBest accuracy : {best_acc}")
# print(f"Number of parameter END: {sizeModel(net)}")

with open('project_results.txt', 'a') as f:
    f.write(f'\n{net.__class__.__name__}; \
            {max_epochs}; \
            {lr}; \
            {best_acc}; \
            {sizeModel(net)}; \
            {args.half}; \
            {args.da}; \
            {pruned_size}; \
            {prune_type}; \
            {args.prune_ratio/100}; \
            {(end-start)/60}; \
            {train_losses}; \
            {test_losses}; \
            {accuracies}; \
            {batch_size}')