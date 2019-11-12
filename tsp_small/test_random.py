import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from gpn import GPN


# args
parser = argparse.ArgumentParser(description="GPN test")
parser.add_argument('--size', default=50, help="size of model")
parser.add_argument('--batch_size', default=1000, help='')
parser.add_argument('--test_size', default=50, help="size of TSP")
parser.add_argument('--test_steps', default=100, help='')
args = vars(parser.parse_args())


B = int(args['batch_size'])
size = int(args['size'])
test_size = int(args['test_size'])
n_test = int(args['test_steps'])

load_root ='./model/gpn_tsp'+str(size)+'.pt'

print('=========================')
print('prepare to test')
print('=========================')
print('Hyperparameters:')
print('size', size)
print('batch size', B)
print('test size', test_size)
print('test steps', n_test)
print('load root:', load_root)
print('=========================')
    
# greedy
model = torch.load(load_root).cuda()

tour_len = 0
total_len = 0


for m in range(n_test):
    tour_len = 0
    
    X = np.random.rand(B, test_size, 2)
    
    X = torch.Tensor(X).cuda()
    
    mask = torch.zeros(B,test_size).cuda()
    
    R = 0
    Idx = []
    reward = 0
    
    Y = X.view(B,test_size,2)           # to the same batch size
    x = Y[:,0,:]
    h = None
    c = None
    
    for k in range(test_size):
        
        output, h, c, _ = model(x=x, X_all=X, h=h, c=c, mask=mask)
        
        idx = torch.argmax(output, dim=1)
        Idx.append(idx.data)
        
        Y1 = Y[[i for i in range(B)], idx.data].clone()
        if k == 0:
            Y_ini = Y1.clone()
        if k > 0:
            reward = torch.norm(Y1-Y0, dim=1)
    
        Y0 = Y1.clone()
        x = Y[[i for i in range(B)], idx.data].clone()
        
        R += reward

        mask[[i for i in range(B)], idx.data] += -np.inf
        
    R += torch.norm(Y1-Y_ini, dim=1)

    tour_len += R.mean().item()

    print('test:{}, total length:{}'.format(m, tour_len))
    
    total_len += tour_len

print('total tour length:', total_len/n_test)
