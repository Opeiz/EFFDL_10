# EFFDL_10


This repository was created to save the changes made in the course taught as UE G "Efficient Deep Learning".

It is based on the original [git](https://github.com/brain-bzh/efficient-deep-learning) of the course and the members of team 10 are :
* [Emmanuela Tertuliano](https://github.com/EmmanuelaTertuliano)
* [Joaquin Opazo](https://github.com/Opeiz)

## Lab Tertuliano and Opazo

This file is the mix of all the other labs, and the labs different values as lab1 or lab2 and others are the specific approach for each class. The main file is [Lab](Lab_TERTULIANO_OPAZO.py)
```
python Lab_TERTULIANO_OPAZO.py [-h] [--model MODEL] [--lr LR] [--epochs EPOCHS] [--alpha ALPHA] [--DataAug DATAAUG]
```
```
usage: Lab_TERTULIANO_OPAZO.py [-h] [--model MODEL] [--lr LR] [--epochs EPOCHS] [--alpha ALPHA] [--DataAug DATAAUG] [--MixUp MIXUP]

Lab EFFDL

options:
  -h, --help         show this help message and exit
  --model MODEL      Options: ResNet18, PreActResNet18, DenseNet121, VGG19
  --lr LR            Learning rate
  --epochs EPOCHS    Number of epochs
  --alpha ALPHA      Alpha value for Mixup
  --DataAug DATAAUG  [Bool] Use Custom Data Augmentation or not
  --MixUp MIXUP      [Bool] Use MixUp or not
```


## Lab 01 Training and Testing

The goal of this lab is the first approach to the Training and Testing code for different models. The main is the [lab1](/lab1.py) and is coded with some arguments, the usage is the following\
```
python lab1.py [-h] [--model MODEL] [--lr LR] [--epochs EPOCHS]
``` 
```
usage: lab1.py [-h] [--model MODEL] [--lr LR] [--epochs EPOCHS]

Lab01 Training/Testing Model

options:
  -h, --help       show this help message and exit
  --model MODEL    Options: ResNet18, PreActResNet18, DenseNet121, VGG19
  --lr LR          Learning rate
  --epochs EPOCHS  Number of epochs
```

## Lab 02 Data Augmentation

The goal of the lab 2 is to play with different type of transforms for data augmentation (DA). The main is the [lab2](/lab2.py)

```
python lab2.py
``` 