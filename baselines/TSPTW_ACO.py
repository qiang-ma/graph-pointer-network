import random
import operator
import matplotlib.pyplot as plt
import math




class Graph(object):
    def __init__(self, cost_matrix: list, rank: int, time):
        """
        :param cost_matrix:
        :param rank: rank of the cost matrix
        """
        self.matrix = cost_matrix
        self.rank = rank
        # noinspection PyUnusedLocal
        self.pheromone = [[1 / (rank * rank) for j in range(rank)] for i in range(rank)]
        self.time = time


class ACO(object):
    def __init__(self, ant_count: int, generations: int, gamma: float, beta: float, rho: float, tilda: float, lamb: float, q: int, q0: float, theta: float, 
                 strategy: int):
        """
        :param ant_count:
        :param generations:
        :param alpha: relative importance of pheromone
        :param beta: relative importance of heuristic information
        :param rho: pheromone residual coefficient
        :param q: pheromone intensity
        :param strategy: pheromone update strategy. 0 - ant-cycle, 1 - ant-quality, 2 - ant-density
        """
        self.Q = q
        self.q0 = q0
        self.rho = rho
        self.beta = beta
        self.theta = theta
        self.gamma = gamma
        self.tilda = tilda
        self.lamb = lamb
        self.ant_count = ant_count
        self.generations = generations
        self.update_strategy = strategy

    def _update_local_pheromone(self, graph: Graph, ants: list):
        for i, row in enumerate(graph.pheromone):
            for j, col in enumerate(row):
                graph.pheromone[i][j] *= 1 - self.rho
                for ant in ants:
                    graph.pheromone[i][j] += self.rho * ant.pheromone_delta[i][j]
    def _update_gloabl_pheromone(self, graph: Graph, best_solution: list, best_cost: float, ants: list):
        for i in range(len(best_solution) - 1):
            graph.pheromone[best_solution[i]][best_solution[i+1]] *= 1 - self.theta
            for ant in ants:
                graph.pheromone[best_solution[i]][best_solution[i+1]] += self.theta / best_cost

    # noinspection PyProtectedMember
    def solve(self, graph: Graph):
        """
        :param graph:
        """
        best_cost = float('inf')
        best_solution = []
        ants = [_Ant(self, graph) for i in range(self.ant_count)]
        fail = False
        for gen in range(self.generations):
            # noinspection PyUnusedLocal
            for ant in ants:
                for i in range(graph.rank - 1):
                    if not ant._select_next():
                      fail = True
#                       for j in range(20):
#                           print(ant.graph.time[j, 0], ant.graph.time[j, 1])
                      break
                ant.total_cost += graph.matrix[ant.tabu[-1]][ant.tabu[0]]
                if ant.total_cost < best_cost:
                    best_cost = ant.total_cost
                    best_solution = [] + ant.tabu
                    print(best_solution, best_cost)
                # update pheromone
                ant._update_pheromone_delta()
            self._update_local_pheromone(graph, ants)
            self._update_gloabl_pheromone(graph, best_solution, best_cost, ants)
            # print('generation #{}, best cost: {}, path: {}'.format(gen, best_cost, best_solution))
        return best_solution, best_cost


class _Ant(object):
    def __init__(self, aco: ACO, graph: Graph):
        self.colony = aco
        self.graph = graph
        self.total_cost = 0.0
        self.tabu = []  # tabu list
        self.pheromone_delta = [[0 for j in range(self.graph.rank)] for i in range(self.graph.rank)]  # the local increase of pheromone
        self.allowed = [i for i in range(graph.rank)]  # nodes which are allowed for the next selection
        self.g = [[0 if i == j else 1 / graph.matrix[i][j] for j in range(graph.rank)] for i in
                    range(graph.rank)]  # arrival information
        self.h = [[0 if i == j else 1 / graph.matrix[i][j] for j in range(graph.rank)] for i in
                    range(graph.rank)]  # entry information
        self.G = np.array([[0 if i == j else 1 / graph.matrix[i][j] for j in range(graph.rank)] for i in
                    range(graph.rank)])  # arrival information
        self.H = np.array([[0 if i == j else 1 / graph.matrix[i][j] for j in range(graph.rank)] for i in
                    range(graph.rank)])  # entry information
        start = 0  # start from any node
        self.tabu.append(start)
        self.current = start
        self.allowed.remove(start)

    def _select_next(self):
        # noinspection PyUnusedLocal
        self._update_heuristics()
        denominator = 0
        for i in self.allowed:
            denominator += self.graph.pheromone[self.current][i]* self.g[self.current][i] ** self.colony.beta  * self.h[self.current][i] ** self.colony.gamma
        if denominator == 0.0:
            return False
        probabilities = [0 for i in range(self.graph.rank)]  # probabilities for moving to a node in the next step
        for i in range(self.graph.rank):
            try:
                self.allowed.index(i)  # test if allowed list contains i
                probabilities[i] = (self.graph.pheromone[self.current][i]* self.g[self.current][i] ** self.colony.beta  * self.h[self.current][i] ** self.colony.gamma) / denominator
                #print(self.graph.pheromone[self.current][i], self.g[self.current][i], self.h[self.current][i])
            except ValueError:
                pass  # do nothing
        # select next node by probability roulette
        selected = 0
        print(probabilities)
        rand = random.random()
        if rand < self.colony.q0:
            selected = np.argmax(np.array(probabilities))
            
        else:
            selected = np.random.choice(np.arange(20), 1, p=probabilities)[0]
        #print(self.total_cost, selected, self.graph.time[selected, 0], self.graph.time[selected, 1])
        self.allowed.remove(selected)
        self.tabu.append(selected)
        self.total_cost += self.graph.matrix[self.current][selected]
        self.total_cost = max(self.total_cost, self.graph.time[selected, 0])
        #print(self.total_cost, selected, self.graph.time[selected, 0], self.graph.time[selected, 1])
        self.current = selected
        return True
    
    def _update_heuristics(self):
        for i in self.allowed:
            self.G[self.current][i] = self.graph.time[i, 1] - (self.total_cost + self.graph.matrix[self.current][i])
            self.H[self.current][i] = self.graph.time[i, 0] - (self.total_cost + self.graph.matrix[self.current][i])
        mean_G = np.mean(self.G[self.current, np.array(self.allowed)])
        mean_H = np.mean(self.H[self.current, np.array(self.allowed)])
        for i in self.allowed:
            self.g[self.current][i] = 1 / (1 + math.exp(self.colony.tilda * (self.G[self.current][i] - mean_G))) if self.G[self.current][i] >= 0 else 0
            self.h[self.current][i] = 1 / (1 + math.exp(self.colony.lamb * (self.H[self.current][i] - mean_H))) if self.H[self.current][i] > 0 else 1
            
    # noinspection PyUnusedLocal
    def _update_pheromone_delta(self):
        for _ in range(1, len(self.tabu)):
            i = self.tabu[_ - 1]
            j = self.tabu[_]
            if self.colony.update_strategy == 1:  # ant-quality system
                self.pheromone_delta[i][j] = self.colony.Q
            elif self.colony.update_strategy == 2:  # ant-density system
                # noinspection PyTypeChecker
                self.pheromone_delta[i][j] = self.colony.Q / self.graph.matrix[i][j]
            else:  # ant-cycle system
                self.pheromone_delta[i][j] = self.colony.Q / self.total_cost


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

        self.data_set = {"x":[], "time":[]}
        
        # randomly sample points uniformly from [0, 1]
        B=num_samples

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
        print(X[:,  0, :2])
        print(X[:,  0, 2:4])
        # shuffle the original sequence
        np.random.shuffle(X[1:])

        X = X.transpose(1,0,2)

        self.solutios = solutions
        self.data_set['x'] = X[:,:,:2]
        self.data_set['time'] = X[:, :, 2:4]
        self.size = len(self.data_set)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data_set["x"][idx], self.data_set["time"][idx]



input_dim = 2
size = 20
train_size = 1000


dataset = TSPDataset(train=True, size=size,
     num_samples=train_size)
from scipy.spatial import distance
import time as tm


start_time = tm.time()
n = 1000
total_len = 0
accuracy = 0
for i in range(n):
    coordinate= dataset.data_set['x'][i, :, :]
    time = dataset.data_set['time'][i, :, :]
    cost_matrix = distance.cdist(coordinate, coordinate, 'euclidean')
    rank = len(cost_matrix)
    aco = ACO(ant_count=10, generations=100, gamma=0.3, beta=0.5, rho=0.1, tilda=0.05, lamb=0.05, q=10, q0=0.99, theta=0.1, strategy=3)
    graph = Graph(cost_matrix, rank, time)
    path, cost = aco.solve(graph)
    if path != []:
        accuracy += 1
        cost += cost_matrix[path[0]][path[-1]]
        total_len += cost
        path.append(path[0])
        
        
    
    print('cost: {}, path: {}, accuracy:{}'.format(cost, path, float(accuracy)))
    coordinate_list = coordinate.tolist()
#     plot(coordinate_list, path)
    
    # time plot
    plot_tot_cost = 0
    for idx, val in enumerate(path):
#         print("point index: ", val, "coordinates: ", coordinate_list[val], "enter time: ", time[val, :], "cuml cost now: ", plot_tot_cost)
        plot_tot_cost += (max(plot_tot_cost+cost_matrix[val, path[idx-1]], time[val, 0]) - plot_tot_cost)
    
end_time = tm.time()
print("total_time: ", end_time - start_time, " avg time: ", (end_time - start_time)/n)
print("avg_cost: ", total_len / accuracy)
