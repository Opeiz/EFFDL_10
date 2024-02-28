# Lab Session 3
# --
# The objectives of this second lab session are the following:
# - Quantize a neural network post-training
# - Quantize during training using Binary Connect
# - Explore the influence of quantization on performance on a modern DL architecture

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

# ## Argument parser
# parser = argparse.ArgumentParser(description='Lab EFFDL')
# parser.add_argument('--model', default='PreActResNet18', type=str, help='Options: ResNet18, PreActResNet18, DenseNet121, VGG19')
# parser.add_argument('--lr', default=0.05, type=float, help='Learning rate')
# parser.add_argument('--epochs', default=5, type=int, help='Number of epochs')
# parser.add_argument('--alpha', default=1.0, type=float, help='Alpha value for Mixup')
# args = parser.parse_args()

# ## Test if there is GPU
device = torch.device("cuda" if torch.cuda.is_available() else "CPU")
# print(device)

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


# ## number of target samples for the final dataset
# num_train_examples = len(c10train)
# num_test_examples = len(c10test)
# num_samples_subset = 15000

# ## We set a seed manually so as to reproduce the results easily
# seed  = 2147483647

# ## Generate a list of shuffled indices ; with the fixed seed, the permutation will always be the same, for reproducibility
# indices = list(range(num_train_examples))
# np.random.RandomState(seed=seed).shuffle(indices)## modifies the list in place

# ## We define the Subset using the generated indices 
# c10train_subset = torch.utils.data.Subset(c10train,indices[:num_samples_subset])
# print(f"Initial CIFAR10 dataset has {len(c10train)} samples")
# print(f"Subset of CIFAR10 dataset has {len(c10train_subset)} samples")

# # Finally we can define anoter dataloader for the training data
# trainloader_subset = DataLoader(c10train_subset,batch_size=32,shuffle=True)

# ## Example of usage of the models
from model_cifar100 import *

# print("\n== Builind the model ==")
# models = {'ResNet18': ResNet18(), 'PreActResNet18': PreActResNet18(), 'DenseNet121': DenseNet121(), 'VGG19': VGG('VGG19')}

# if args.model in models:
#     print(f"\nModel found in the list, going to train :)")
#     print(f"Training {args.model} model")
#     model = models.get(args.model)
# else:
#     model = models.get('PreActResNet18')

# net = model
# net = net.to(device)
# net = net.half() ## Quantization of the modelss

# ## Hyperparameters to set
# lr = args.lr
# criterion = nn.CrossEntropyLoss()
# max_epochs = args.epochs
# best_acc = 0
# optimizer = optim.SGD(net.parameters(), lr=lr)
# alpha = args.alpha
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# ## Save results
# results = []
# params = []
# train_losses = []
# test_losses = []

# # Testing the model
# def test(epoch):
    
#     global best_acc
#     net.eval()

#     test_loss = 0
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(testloader):
#             inputs, targets = inputs.to(device), targets.to(device)
#             inputs = inputs.half()  ## Quantization of the inputs
#             outputs = net(inputs)
#             loss = criterion(outputs, targets)

#             test_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()
    
#     test_losses.append(test_loss / len(testloader))
#     print(f"Test Loss: {test_loss/(batch_idx+1)}")

#     # Save checkpoint.
#     acc = 100.*correct/total
#     if acc > best_acc:
#         state = {
#             'net': net.state_dict(),
#             'acc': acc,
#             'epoch': epoch,
#         }
#         pathckpt = f'./ckpts/ckpt_{args.model}.pth'
#         torch.save(state, pathckpt)
#         best_acc = acc

# # Function to count the parameters in the model
# def contar_parametros(modelo):
#     return sum(p.numel() for p in modelo.parameters() if p.requires_grad)

start = time.time()

## Load the best model
loaded_cpt = torch.load('ckpts/ckpt_PreActResNet18.pth')

# Define the model 
model = PreActResNet18()  # Assuming the model is PreActResNet18
model = model.to(device)
# Finally we can load the state_dict in order to load the trained parameters 
model.load_state_dict(loaded_cpt['net'])

# If you use this model for inference (= no further training), you need to set it into eval mode
model.eval()

optimizer = optim.SGD(model.parameters(), lr=0.05)

## Quantization of the model
correct = 0
total = 0
test_loss = 0
model = model.half()  # convert model to half precision
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs.half()  ## Quantization of the inputs
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    acc = 100.*correct/total
    print(f"Accuracy of the model: {acc}")


# print(f"Best accuracy: {best_acc}")
# print(f"Number of parameters in the model: {contar_parametros(model)}")

end = time.time()
print(f"\nTime to train the model: {(end-start)/60} minutes")

# with open('results_lab3.txt', 'a') as f:
#     f.write(f'\n{args.model}, {max_epochs}, {lr}, {best_acc}, {contar_parametros(net)}, {(end-start)/60}' )

## Plotting the results
# import matplotlib.pyplot as plt

# epochs = range(1, max_epochs+1)

# plt.plot(epochs, train_losses, label='Training Loss')
# plt.plot(epochs, test_losses, label='Test Loss')
# plt.title('Loss Plotting for {model} model')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
# save_path = f'images/Lab3/loss_plot_{max_epochs}_{lr}_half.png'
# plt.savefig(save_path)


