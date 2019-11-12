import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from gpn import GPN
from tqdm import tqdm
import numpy as np


# Test Data
class TSPDataset(Dataset):
    
    def __init__(self, dataset_fname=None, train=False, size=50, num_samples=100000, random_seed=1111):
        super(TSPDataset, self).__init__()
        
        torch.manual_seed(random_seed)
        
        self.data_set = []
        
        # randomly sample points uniformly from [0, 1]
        for l in tqdm(range(num_samples)):
            x = torch.FloatTensor(2, size).uniform_(0, 1)
            self.data_set.append(x)
        
        self.size = len(self.data_set)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data_set[idx]

# args
parser = argparse.ArgumentParser(description="GPN test")
parser.add_argument('--size', default=50, help="size of model")
args = vars(parser.parse_args())
size = int(args['size'])

load_data = './data/test_tsp'+str(size)+'.pt'
test_data = torch.load(load_data)

# there are 1000 test data
X = torch.zeros(1000, size, 2)
for i in range(1000):
    for j in range(2):
        X[i,:,j] = test_data[i][j]

Z = X.view(4, 250, size, 2).clone()

load_root ='./model/gpn_tsp'+str(size)+'.pt'

print('=========================')
print('test for TSP'+str(size))
print('=========================')


model = torch.load(load_root).cuda()

# greedy
B = 250
total_tour_len = 0

for i in range(4):
    
    tour_len = 0
    
    X = Z[i].cuda()
    
    mask = torch.zeros(B, size).cuda()
    
    R = 0
    Idx = []
    reward = 0
    
    Y = X.view(B, size, 2)           # to the same batch size
    x = Y[:,0,:]
    h = None
    c = None
    
    for k in range(size):
        
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
    print('tour length:', tour_len)
    total_tour_len += tour_len

print('total tour length:', total_tour_len/4)
