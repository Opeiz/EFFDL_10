    # Aware quantization to binary
from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch 
import torchvision
import torch.optim as optim
from torch import nn
from model_cifar100 import *
import numpy as np
from binnaryconnect import *
import time

## Normalization adapted for CIFAR10
normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

## Test if there is GPU
device = torch.device("cuda" if torch.cuda.is_available() else "CPU")
print(device)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize_scratch,
])
rootdir = './data/cifar10'

c10train = CIFAR10(rootdir,train=True,download=True,transform=transform_train)
c10test = CIFAR10(rootdir,train=False,download=True,transform=transform_test)
trainloader = DataLoader(c10train,batch_size=32,shuffle=True)
testloader = DataLoader(c10test,batch_size=32)
num_train_examples = len(c10train)
num_samples_subset = 1500
train_losses = []
test_losses = []
## We set a seed manually so as to reproduce the results easily
seed  = 2147483647

## Generate a list of shuffled indices ; with the fixed seed, the permutation will always be the same, for reproducibility
indices = list(range(num_train_examples))
np.random.RandomState(seed=seed).shuffle(indices)## modifies the list in place

## We define the Subset using the generated indices 
c10train_subset = torch.utils.data.Subset(c10train,indices[:num_samples_subset])
trainloader_subset = DataLoader(c10train_subset,batch_size=32,shuffle=True)

loaded_cpt = torch.load('ckpts/ckpt_PreActResNet18_[1111].pth')

# Define the model 
model = PreActResNet18()  # Assuming the model is PreActResNet18

# Finally we can load the state_dict in order to load the trained parameters 
model.load_state_dict(loaded_cpt['net'])

netBC = BC(model)
netBC.model = netBC.model.to(device)
## Hyperparameters to set
lr = 0.05
criterion = nn.CrossEntropyLoss()
max_epochs = 25
best_acc = 0
optimizer = optim.SGD(netBC.model.parameters(), lr=lr)


## Training the model
def train(epoch):
    epch_train_loss = []
    print('\nEpoch: %d' % epoch)
    netBC.model.train()
    train_loss = 0
    correct = 0
    total = 0
    count = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        netBC.binarization()
        outputs = netBC.forward(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        netBC.restore()
        optimizer.step()
        netBC.clip()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        count+=1
    train_losses.append(train_loss / len(trainloader))
    print(f"Train Loss: {train_loss/(batch_idx+1)}")
        
# Testing the model
def test(epoch):
    
    global best_acc
    netBC.model.eval()

    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = netBC.model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    test_losses.append(test_loss / len(testloader))
    print(f"Test Loss: {test_loss/(batch_idx+1)}")  
    acc = 100.*correct/total
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        state = {
            'net': netBC.model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        pathckpt = f'./ckpts/ckpt_PreActResNet18_[1111]_BN.pth'
        torch.save(state, pathckpt)
        best_acc = acc

# Function to count the parameters in the model
def contar_parametros(modelo):
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)

start = time.time()

for i in range(max_epochs):
    train(i)
    test(i)

epochs = range(1, max_epochs+1)

end = time.time()
# print(f"\nTime to train the model: {(end-start)/60} minutes")

with open('results_presentation.txt', 'a') as f:
    f.write(f'\n{"PreActResNet18"}; {max_epochs}; {lr}; {best_acc}; {contar_parametros(netBC.model)}; {(end-start)/60}; {0}; {train_losses}; {test_losses}' )


plt.plot(epochs, train_losses, label='Training Loss')
plt.plot(epochs, test_losses, label='Test Loss')
plt.title('Loss Plotting')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig("images/loss_plot-awareQuantization.png")
