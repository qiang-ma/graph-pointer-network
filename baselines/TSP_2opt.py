from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import numpy as np
import time
from scipy.spatial import distance


# Training Data
class TSPDataset(Dataset):
    def __init__(self, dataset_fname=None, train=False, size=50, num_samples=100000, random_seed=1111):
        super(TSPDataset, self).__init__()
        torch.manual_seed(random_seed)
        self.data_set = []

        # randomly sample points uniformly from [0, 1]
        for l in tqdm(range(num_samples)):
            x = torch.FloatTensor(2, size).uniform_(0, 1)
            # x = torch.cat([start, x], 1)
            self.data_set.append(x)

        self.size = len(self.data_set)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data_set[idx]

# define problem dimension
input_dim = 2
# define tsp size
size = 50
# define training size
train_size = 1000

dataset = TSPDataset(train=True, size=size,
     num_samples=train_size)

# save the dataset
torch.save(dataset, './TSP50_1000.pt')
# load the dataset
dataset = torch.load('./TSP50_1000.pt')

# Run TSP nearest neighbor search
start = time.time()
avg_dist = 0


# 2OPT
def cost(route, distance_matrix):
    return distance_matrix[[route[-1:] + route[:-1]], route].sum()


def cost_change(cost_mat, n1, n2, n3, n4):
    result = cost_mat[n1, n3] + cost_mat[n2, n4] - cost_mat[n1, n2] - cost_mat[n3, n4]
    return result


def two_opt(route, distance_matrix, size):
    best = route
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route)-2):
            for j in range(i+1, len(route)):
                if j-i == 1: continue # changes nothing, skip then
                if cost_change(distance_matrix, best[i - 1], best[i], best[j - 1], best[j]) < -1e-9:
                    best[i:j] = best[j - 1:i - 1:-1]
                    improved = True
        route = best
    return route, cost(route, distance_matrix)


start = time.time()
route = [x for x in range(size)] + [0]
avg_cost = 0
for i in range(train_size):
    graph = dataset.__getitem__(i).numpy().T
    distance_matrix = distance.cdist(graph, graph, 'euclidean')
    best_route, tot_cost = two_opt(route, distance_matrix, size)
    avg_cost += tot_cost
#     print(i, avg_cost / (i+1), time.time() - start)
end = time.time()
print(str(train_size) + " graph used time: ", end - start)
print(str(train_size) + " graph avg dist: ", avg_cost / train_size)

