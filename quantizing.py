# Aware quantization to binary
import matplotlib.pyplot as plt
import torch 
import torch.optim as optim
import numpy as np
import time
from binnaryconnect import *
from model_cifar100 import *
  
train_losses = []
test_losses = []
best_acc = 0

def halfquantizing(net,ckpt_file,criterion,max_epochs,best_acc,testloader,device):
    loaded_cpt = torch.load(ckpt_file)
    
    #load the state_dict in order to load the trained parameters 
    net.load_state_dict(loaded_cpt['net'])
    #ckpt = torch.load(ckpt_file)
    #net = ckpt['net']
    net = net.half()
    print("ckpt")
    #testing
    for epoch in range(max_epochs):
        net.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.half()
                outputs = net(inputs)
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
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            pathckpt = ckpt_file.replace(".","_half.")
            torch.save(state, pathckpt)
            best_acc = acc

    
    return train_losses,test_losses
        
def binaryQuantizing(net,ckpt_file,criterion,max_epochs,optimizer,trainloader,testloader,device,fact_factor):
    netBC = BC(net)
    netBC.model = netBC.model.to(device)
    ## Training the model
    for epoch in range(max_epochs):
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

    ## Testing the model
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
            if fact_factor != 0:
                pathckpt = ckpt_file.replace(".",f"{fact_factor}_binnary.")
            else:
                pathckpt = ckpt_file.replace(".","_binnary.")
            torch.save(state, pathckpt)
            best_acc = acc
    return train_losses,test_losses

