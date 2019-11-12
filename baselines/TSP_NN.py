from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import numpy as np
import time


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

for j in range(train_size):
    graph = dataset.__getitem__(j).numpy()
    a = graph[:, 0].reshape(2, 1)
    route = [0]
    points_left = [x for x in range(1, size)]
    total_dist = 0
    for i in range(size-1):
        # Calculate distance from one point to every other point
        dist = np.sqrt(np.square(graph[0, :] - a[0]) + (np.square(graph[1, :] - a[1])))
        dist = np.vstack((np.linspace(0, size-1, num=size).reshape(1, size), dist))
        filtered = dist[:, np.array(points_left)]
        closest_idx = np.argmin(filtered[1, :])
        real_idx = int(filtered[0, closest_idx])
        total_dist += dist[1, real_idx]
        a = graph[:, real_idx].reshape(2, 1)
        route.append(real_idx)
        points_left.remove(real_idx)
    total_dist += dist[1, 0]

    avg_dist += 1 / train_size * total_dist
end = time.time()
print(str(train_size) + " graph used time: ", end - start)
print(str(train_size) + " graph avg dist: ", avg_dist)

