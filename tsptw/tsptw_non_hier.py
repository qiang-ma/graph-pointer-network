import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from tqdm import tqdm
from gpn import GPN
from scipy.spatial import distance


'''
generate training data
'''
def generate_data(B=512, size=50):

    graph = np.random.rand(size, B, 2)
    X = np.zeros([size, B, 4])  # xi, yi, ei, li, ci
    solutions = np.zeros(B) 
    route = [x for x in range(size)] + [0]
    total_tour_len = 0
    
    for b in range(B):
        best = route.copy()
        # begin 2-opt
        graph_ = graph[:,b,:].copy()
            
        dmatrix = distance.cdist(graph_, graph_, 'euclidean')
        improved = True
        
        while improved:
            improved = False
            
            for i in range(size):
                for j in range(i+2, size+1):
                    
                    old_dist = dmatrix[best[i],best[i+1]] + dmatrix[best[j], best[j-1]]
                    new_dist = dmatrix[best[j],best[i+1]] + dmatrix[best[i], best[j-1]]
                    
                    # new_dist = 1000
                    if new_dist < old_dist:
                        best[i+1:j] = best[j-1:i:-1]
                        # print(opt_tour)
                        improved = True  
    
        cur_time = 0
        tour_len = 0
        X[0,b,:2] = graph_[best[0]]  # x0, y0
        X[0,b,2] = 0  # e0 = 0
        X[0,b,3] = 2*np.random.rand(1)  # l0 = rand
        #  X[0,b,4] = 0
        
        for k in range(1, size):
            # generate data with approximate solutions
            X[k,b,:2] = graph_[best[k]]   # xi, yi
            cur_time += dmatrix[best[k-1], best[k]]
            tour_len += dmatrix[best[k-1], best[k]]
            X[k,b,2] = np.max([0, cur_time - 2*np.random.rand(1)])  # entering time 0<= ei <= cur_time
            X[k,b,3] = cur_time + 2*np.random.rand(1) + 1  # leaving time li >= cur_time
            # X[k,b,4] = cur_time    # indicate the optimal solution
        tour_len += dmatrix[best[size-1], best[size]]    
        solutions[b] += tour_len

    # shuffle the original sequence
    np.random.shuffle(X)
    
    X = X.transpose(1,0,2)

    return X, solutions


'''
main
'''

parser = argparse.ArgumentParser(description="GPN with RL")
parser.add_argument('--size', default=20, help="size of TSPTW")
parser.add_argument('--epoch', default=20, help="number of epochs")
parser.add_argument('--batch_size', default=512, help='')
parser.add_argument('--train_size', default=2500, help='')
parser.add_argument('--val_size', default=1000, help='')
parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
args = vars(parser.parse_args())

size = int(args['size'])
learn_rate = args['lr']    # learning rate
B = int(args['batch_size'])    # batch_size
B_val = int(args['val_size'])    # validation size
steps = int(args['train_size'])    # training steps
n_epoch = int(args['epoch'])    # epochs
save_root ='./model/gpn_tsptw'+str(size)+'.pt'

print('=========================')
print('prepare to train')
print('=========================')
print('Hyperparameters:')
print('size', size)
print('learning rate', learn_rate)
print('batch size', B)
print('validation size', B_val)
print('steps', steps)
print('epoch', n_epoch)
print('save root:', save_root)
print('=========================')

model = GPN(n_feature=4, n_hidden=128).cuda()
optimizer = optim.Adam(model.parameters(), lr=learn_rate)

lr_decay_step = 2500
lr_decay_rate = 0.96
opt_scheduler = lr_scheduler.MultiStepLR(optimizer, range(lr_decay_step, lr_decay_step*1000,
                                                          lr_decay_step), gamma=lr_decay_rate)

beta = 0.8
val_mean = []
val_std = []
val_accuracy = []


for epoch in range(n_epoch):
    for i in tqdm(range(steps)):
        
        optimizer.zero_grad()
    
        X, solutions = generate_data(B=B, size=size)
        Enter = X[:,:,2]   # Entering time
        Leave = X[:,:,3]   # Leaving time
        
        X = torch.Tensor(X).cuda()
        Enter = torch.Tensor(Enter).cuda()
        Leave = torch.Tensor(Leave).cuda()
        mask = torch.zeros(B,size).cuda()
    
        R = 0
        logprobs = 0
        reward = 0
        
        time_wait = torch.zeros(B).cuda()
        time_penalty = torch.zeros(B).cuda()
        total_time_penalty = torch.zeros(B).cuda()
        total_time_cost = torch.zeros(B).cuda()
        total_time_wait = torch.zeros(B).cuda()

        
        # X = X.view(B,size,3)
        # Time = Time.view(B,size)

        x = X[:,0,:]
        h = None
        c = None
    
        for k in range(size):
        
            output, h, c, _ = model(x=x, X_all=X, h=h, c=c, mask=mask)
            sampler = torch.distributions.Categorical(output)
            idx = sampler.sample()         # now the idx has B elements
    
            y_cur = X[[i for i in range(B)], idx.data].clone()
            if k == 0:
                y_ini = y_cur.clone()
            if k > 0:
                reward = torch.norm(y_cur[:,:2] - y_pre[:,:2], dim=1)
                
            y_pre = y_cur.clone()
            x = X[[i for i in range(B)], idx.data].clone()
            
            R += reward
            total_time_cost += reward
            
            # enter time
            enter = Enter[[i for i in range(B)], idx.data]
            leave = Leave[[i for i in range(B)], idx.data]
            
            # determine the total reward and current enter time
            time_wait = torch.lt(total_time_cost, enter).float()*(enter - total_time_cost)  
            total_time_wait += time_wait     # total time cost
            total_time_cost += time_wait
            
            time_penalty = torch.lt(leave, total_time_cost).float()*10
            total_time_cost += time_penalty
            total_time_penalty += time_penalty

        
            TINY = 1e-15
            logprobs += torch.log(output[[i for i in range(B)], idx.data]+TINY) 
            
            mask[[i for i in range(B)], idx.data] += -np.inf 

        R += torch.norm(y_cur[:,:2] - y_ini[:,:2], dim=1)
        total_time_cost += torch.norm(y_cur[:,:2] - y_ini[:,:2], dim=1)

        if i == 0:
            C = total_time_cost.mean()
        else:
            C = (total_time_cost * beta) + ((1. - beta) * total_time_cost.mean())
        
        loss = ((total_time_cost - C)*logprobs).mean()
    
        loss.backward()
        
        max_grad_norm = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_grad_norm, norm_type=2)
        optimizer.step()
        opt_scheduler.step()

        if i % 50 == 0:
            print("epoch:{}, batch:{}/{},  total time:{}, reward:{}, time:{}"
                .format(epoch, i, steps, total_time_cost.mean().item(),
                        R.mean().item(), total_time_wait.mean().item()))
            # R_mean.append(R.mean().item())
            # R_std.append(R.std().item())

            print("optimal upper bound:{}"
                  .format(solutions.mean()))
            
            X, solutions = generate_data(B=B_val, size=size)
            Enter = X[:,:,2]   # Entering time
            Leave = X[:,:,3]   # Leaving time
            
            X = torch.Tensor(X).cuda()
            Enter = torch.Tensor(Enter).cuda()
            Leave = torch.Tensor(Leave).cuda()
            mask = torch.zeros(B_val, size).cuda()
        
            baseline = 0
            time_wait = torch.zeros(B_val).cuda()
            time_penalty = torch.zeros(B_val).cuda()
            total_time_cost = torch.zeros(B_val).cuda()
            total_time_penalty = torch.zeros(B_val).cuda()

            x = X[:,0,:]
            h = None
            c = None
        
            for k in range(size):
            
                output, h, c, _ = model(x=x, X_all=X, h=h, c=c, mask=mask)
                idx = torch.argmax(output, dim=1)    # greedy baseline
        
                y_cur = X[[i for i in range(B_val)], idx.data].clone()
                if k == 0:
                    y_ini = y_cur.clone()
                if k > 0:
                    baseline = torch.norm(y_cur[:,:2] - y_pre[:,:2], dim=1)
                    
                y_pre = y_cur.clone()
                x = X[[i for i in range(B_val)], idx.data].clone()

                total_time_cost += baseline
                
                # enter time
                enter = Enter[[i for i in range(B_val)], idx.data]
                leave = Leave[[i for i in range(B_val)], idx.data]
                
                # determine the total reward and current enter time
                time_wait = torch.lt(total_time_cost, enter).float()*(enter - total_time_cost)  
                total_time_cost += time_wait
                
                time_penalty = torch.lt(leave, total_time_cost).float()*10
                total_time_cost += time_penalty
                total_time_penalty += time_penalty
                
                mask[[i for i in range(B_val)], idx.data] += -np.inf 
            total_time_cost += torch.norm(y_cur[:,:2] - y_ini[:,:2], dim=1)
            accuracy = 1 - torch.lt(torch.zeros_like(total_time_penalty), total_time_penalty).sum().float() / total_time_penalty.size(0)
            print('validation result:{}, accuracy:{}'
                  .format(total_time_cost.mean().item(), accuracy))
            
            val_mean.append(total_time_cost.mean().item())
            val_std.append(total_time_cost.std().item())
            val_accuracy.append(accuracy)
                  
    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, save_root)
    
val_mean_tsptw = np.array(val_mean)
val_std_tsptw = np.array(val_std)
val_accuracy = np.array(val_accuracy)
np.save('./plot/val_mean_tsptw_'+str(size), val_mean_tsptw)
np.save('./plot/val_std_tsptw_'+str(size), val_std_tsptw)
np.save('./plot/val_accuracy_'+str(size), val_accuracy)

