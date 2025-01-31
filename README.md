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

## Long Project results

Our approach for the long project of Efficient Deep Learning it is based in the [Lab file](LongProject.py) where is the base code for the training and testing of the model we choose to implemente "PreActResNet18()". 

### How to Use

```
usage: LongProject.py [-h] [--lr LR] [--epochs EPOCHS] [--epochs_fine EPOCHS_FINE] [--batch BATCH] [--da] [--prune] [--prune_ratio PRUNE_RATIO] [--fine_tuning] [--custom] [--structured] [--unstructured] [--globalprune] [--half] [--binnary] [--factorized] [--factor FACTOR] [--ckpt CKPT] [--ckptname CKPTNAME]
```

```
Project EFFDL

optional arguments:
  -h, --help            show this help message and exit
  --lr LR               Learning rate
  --epochs EPOCHS       Number of epochs
  --epochs_fine EPOCHS_FINE
                        Number of epochs for fine tuning
  --batch BATCH         Batch size
  --da                  Use data augmentation or not
  --prune               Prune the model or not
  --prune_ratio PRUNE_RATIO
                        Amount for Pruning
  --fine_tuning         Use fine tuning or not
  --custom              Use custom pruning or not
  --structured          Use structured pruning or not
  --unstructured        Use unstructured pruning or not
  --globalprune         Use global pruning or not
  --half                Use half precision or not
  --binnary             Binnary Aware Quantization
  --factorized          Model to use
  --factor FACTOR       Factor for the factorized model
  --ckpt CKPT           Path to the checkpoint
  --ckptname CKPTNAME   Name of the checkpoint
```

### Plot the results

For the results we implement a .txt file that we will read will pandas to see it as a .csv table. The code is [HERE](pandas.ipynb)

