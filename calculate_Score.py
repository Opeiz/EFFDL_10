import os
import torch
import torch.nn as nn
#import resnet
from model_cifar100 import *
from model_cifar100.preact_resnet_factorized import PreActResNet18_fact
import argparse
parser = argparse.ArgumentParser(description='Project EFFDL')

# Classic arguments for the model
parser.add_argument('--ckpt', type=str)
parser.add_argument('--quant', default= 32, type=int)
parser.add_argument('--prune_ratio', default= 0, type=float)
parser.add_argument('--fact_ratio', default= 0, type=int)
args = parser.parse_args()

#ckpt_file = str(input("ckpt file name:"))
#our_quant=int(input("Insert the precision(1,8,16 or 32): "))
quant_factors={1:32,8:4,16:2,32:1}
quant_factor= quant_factors[args.quant]
sparsity= args.prune_ratio
#factorize = int(input("Insert the factor(1 or 2) of factorization used, 0 if not factorized: "))
factorize = args.fact_ratio
def count_conv2d(m, x, y):
    x = x[0] # remove tuple

    fin = m.in_channels
    fout = m.out_channels
    sh, sw = m.kernel_size

    # ops per output element
    kernel_mul = sh * sw * fin
    kernel_add = sh * sw * fin - 1
    bias_ops = 1 if m.bias is not None else 0
    kernel_mul = kernel_mul/quant_factor
    ops = (kernel_mul + kernel_add)/m.groups + bias_ops

    # total ops
    num_out_elements = y.numel()
    total_ops = num_out_elements * ops
   # total_ops = total_ops*(1-sparsity)
   # total_params = int(m.total_params.item()*(1-sparsity))

    #Nice Formatting
    print("{:<10}: S_c={:<4}, F_in={:<4}, F_out={:<4}, P={:<5}, params={:<10}, operations={:<20}".format("Conv2d",sh,fin,fout,x.size()[2:].numel(),int(m.total_params.item()*(1-sparsity)),int(total_ops)))
    # print("Conv2d: S_c={}, F_in={}, F_out={}, P={}, params={}, operations={}".format(sh,fin,fout,x.size()[2:].numel(),int(m.total_params.item()),int(total_ops)))
    # incase same conv is used multiple times
    m.total_ops += torch.Tensor([int(total_ops)])


def count_bn2d(m, x, y):
    x = x[0] # remove tuple

    nelements = x.numel()
    total_sub = 2*nelements
    total_div = nelements
    total_ops = total_sub + total_div

    m.total_ops += torch.Tensor([int(total_ops)])
    #Nice Formatting
    print("{:<10}: S_c={:<4}, F_in={:<4}, F_out={:<4}, P={:<5}, params={:<10}, operations={:<20}".format("Batch norm",'x',x.size(1),'x',x.size()[2:].numel(),int(m.total_params.item()),int(total_ops)))
    # print("Batch norm: F_in={} P={}, params={}, operations={}".format(x.size(1),x.size()[2:].numel(),int(m.total_params.item()),int(total_ops)))


def count_relu(m, x, y):
    x = x[0]

    nelements = x.numel()
    total_ops = nelements

    m.total_ops += torch.Tensor([int(total_ops)])
    print("ReLU: F_in={} P={}, params={}, operations={}".format(x.size(1),x.size()[2:].numel(),0,int(total_ops)))



def count_avgpool(m, x, y):
    x = x[0]
    total_add = torch.prod(torch.Tensor([m.kernel_size])) - 1
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])
    print("AvgPool: S={}, F_in={}, P={}, params={}, operations={}".format(m.kernel_size,x.size(1),x.size()[2:].numel(),0,int(total_ops)))

def count_linear(m, x, y):
    # per output element
    total_mul = m.in_features/quant_factor
    total_add = m.in_features - 1
    num_elements = y.numel()
    total_ops = (total_mul + total_add) * num_elements
    #total_ops = total_ops*(1-sparsity)
    #total_params = int(m.total_params.item()*(1-sparsity))
    print("Linear: F_in={}, F_out={}, params={}, operations={}".format(m.in_features,m.out_features, int(m.total_params.item()*(1-sparsity)),int(total_ops)))
    m.total_ops += torch.Tensor([int(total_ops)])

def count_sequential(m, x, y):
    print ("Sequential: No additional parameters  / op")

# custom ops could be used to pass variable customized ratios for quantization
def profile(model, input_size, custom_ops = {}):

    model.eval()

    def add_hooks(m):
        if len(list(m.children())) > 0: return
        m.register_buffer('total_ops', torch.zeros(1))
        m.register_buffer('total_params', torch.zeros(1))

        for p in m.parameters():
            m.total_params += torch.Tensor([p.numel()]) / quant_factor # Score changes with quantification

        if isinstance(m, nn.Conv2d):
            m.register_forward_hook(count_conv2d)
        elif isinstance(m, nn.BatchNorm2d):
            m.register_forward_hook(count_bn2d)
        elif isinstance(m, nn.ReLU):
            m.register_forward_hook(count_relu)
        elif isinstance(m, (nn.AvgPool2d)):
            m.register_forward_hook(count_avgpool)
        elif isinstance(m, nn.Linear):
            m.register_forward_hook(count_linear)
        elif isinstance(m, nn.Sequential):
            m.register_forward_hook(count_sequential)
        else:
            print("Not implemented for ", m)

    model.apply(add_hooks)

    x = torch.zeros(input_size)
    model(x)

    total_ops = 0
    total_params = 0
    for m in model.modules():
        if len(list(m.children())) > 0: continue
        if isinstance(m, nn.Conv2d) | isinstance(m, nn.Linear):
            total_ops += m.total_ops*(1-sparsity)
            total_params += m.total_params*(1-sparsity)
        else:
            total_ops += m.total_ops
            total_params += m.total_params

    return total_ops, total_params

def main():
    # Resnet18 - Reference for CIFAR 10
    ref_params = 5586981
    ref_flops  = 834362880

    #PreActResnet18 - [2222] calculate using thop library
    #ref_params = 11217316
    #ref_flops  = 557664256
    loaded_cpt = torch.load(f'ckpts/project/{args.ckpt}')

    # Define the model 
    model = PreActResNet18_fact(factorize)  # Assuming the model is PreActResNet18

    # Finally we can load the state_dict in order to load the trained parameters 
    model.load_state_dict(loaded_cpt['net'])

    #model = resnet.ResNet18()
    #print(model)
    flops, params = profile(model, (1,3,32,32))
    flops, params = flops.item(), params.item()

    score_flops = flops / ref_flops
    score_params = (params / ref_params)
    score = score_flops + score_params
    print("Flops: {}, Params: {}".format(flops,params))
    print("Score flops: {} Score Params: {}".format(score_flops,score_params))
    print("Final score: {}".format(score))

    with open('scores.txt', 'a') as f:
        f.write(f'\n{args.ckpt};{args.quant};{args.prune_ratio};{args.fact_ratio};{score};{score_params};{score_flops};{params}')    

if __name__ == "__main__":
    main()
