from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import time
from scipy.spatial import distance


# Training Data
class TSPDataset(Dataset):

    def __init__(self, size=50, num_samples=100000, random_seed=1111):
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


def create_data_model(graph, vehicles, depot):
    data = dict()
    data['distance_matrix'] = (distance.cdist(graph, graph, 'euclidean') * (10 ** 9)).tolist()
    data['num_vehicles'] = vehicles
    data['depot'] = depot
    return data


def print_solution(manager, routing, assignment):
    """Prints assignment on console."""
    print('Objective: {}'.format(assignment.ObjectiveValue() / 10 ** 9))
    index = routing.Start(0)
    plan_output = 'Route for vehicle 0:\n'
    route_distance = 0
    while not routing.IsEnd(index):
        plan_output += ' {} ->'.format(manager.IndexToNode(index))
        previous_index = index
        index = assignment.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    plan_output += ' {}\n'.format(manager.IndexToNode(index))
    print(plan_output)
    plan_output += 'Route distance: {}miles\n'.format(route_distance)


def main(graph):
    """Entry point of the program."""
    # Instantiate the data problem.
    data = create_data_model(graph, 1, 0)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(data['distance_matrix']), data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.SAVINGS)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = 5

    # Solve the problem.
    assignment = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if assignment:
        print_solution(manager, routing, assignment)
        return assignment.ObjectiveValue()


if __name__ == '__main__':
    input_dim = 2
    size = 50
    train_size = 10

    dataset = TSPDataset(size=size, num_samples=train_size)

    start = time.time()
    avg_cost = 0
    for i in range(train_size):
        graph = dataset.__getitem__(i).numpy().T
        assignment = main(graph)
        avg_cost += assignment
    end = time.time()
    print("Time spent: ", end - start)
    print("Average distance:", avg_cost / train_size / 10 ** 9)

