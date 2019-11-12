from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from scipy.spatial import distance
import time as tm
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


# generate TSPTW dataset
def generate_data(B=512, size=50):
    graph = np.random.rand(size, B, 2)
    X = np.zeros([size, B, 4])  # xi, yi, ei, li, ci
    solutions = np.zeros(B)
    route = [x for x in range(size)] + [0]
    total_tour_len = 0

    for b in range(B):
        best = route.copy()
        # begin 2-opt
        graph_ = graph[:, b, :].copy()

        dmatrix = distance.cdist(graph_, graph_, 'euclidean')
        improved = True

        while improved:
            improved = False

            for i in range(size):
                for j in range(i + 2, size + 1):

                    old_dist = dmatrix[best[i], best[i + 1]] + dmatrix[best[j], best[j - 1]]
                    new_dist = dmatrix[best[j], best[i + 1]] + dmatrix[best[i], best[j - 1]]

                    if new_dist < old_dist:
                        best[i + 1:j] = best[j - 1:i:-1]
                        improved = True

        cur_time = 0
        tour_len = 0
        X[0, b, :2] = graph_[best[0]]  # x0, y0
        X[0, b, 2] = 0  # e0 = 0
        X[0, b, 3] = 2 * np.random.rand(1)  # l0 = rand

        for k in range(1, size):
            # generate data with approximate solutions
            X[k, b, :2] = graph_[best[k]]  # xi, yi
            cur_time += dmatrix[best[k - 1], best[k]]
            tour_len += dmatrix[best[k - 1], best[k]]
            X[k, b, 2] = np.max([0, cur_time - 2 * np.random.rand(1)])  # entering time 0<= ei <= cur_time
            X[k, b, 3] = cur_time + 2 * np.random.rand(1) + 1  # leaving time li >= cur_time
        tour_len += dmatrix[best[size - 1], best[size]]
        solutions[b] += tour_len

    # shuffle the original sequence
    np.random.shuffle(X[1:])
    X = X.transpose(1, 0, 2)

    return X, solutions


def create_data_model(graph, time):
    graph, time = graph, time * (10 ** 6)
    data = dict()
    data['time_matrix'] = (distance.cdist(graph, graph, 'euclidean') * (10 ** 6))
    data['time_windows'] = time
    data['num_vehicles'] = 1
    data['depot'] = 0
    return data


def main(sample_size):
    """Solve the VRP with time windows."""
    start = tm.time()
    avg_length = 0
    count = 0
    for graphid in range(sample_size):
        graph, time = dataset[0][graphid, :, :2], dataset[0][graphid, :, 2:]
        data = create_data_model(graph, time)
        manager = pywrapcp.RoutingIndexManager(
            len(data['time_matrix']), data['num_vehicles'], data['depot'])
        routing = pywrapcp.RoutingModel(manager)

        def time_callback(from_index, to_index):
            """Returns the travel time between the two nodes."""
            # Convert from routing variable Index to time matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['time_matrix'][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(time_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        time = 'Time'
        routing.AddDimension(
            transit_callback_index,
            30 * 10 ** 6,  # allow waiting time
            30 * 10 ** 6,  # maximum time per vehicle
            False,  # Don't force start cumul to zero.
            time)
        time_dimension = routing.GetDimensionOrDie(time)
        # Add time window constraints for each location except depot.
        for location_idx, time_window in enumerate(data['time_windows']):
            if location_idx == 0:
                continue
            index = manager.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(int(time_window[0]), int(time_window[1]))
        # Add time window constraints for each vehicle start node.
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            time_dimension.CumulVar(index).SetRange(int(data['time_windows'][0][0]),
                                                    int(data['time_windows'][0][1]))
        for i in range(data['num_vehicles']):
            routing.AddVariableMinimizedByFinalizer(
                time_dimension.CumulVar(routing.Start(i)))
            routing.AddVariableMinimizedByFinalizer(
                time_dimension.CumulVar(routing.End(i)))
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.SAVINGS)
        search_parameters.time_limit.seconds = 1000
        assignment = routing.SolveWithParameters(search_parameters)
        if assignment:
            time_dimension = routing.GetDimensionOrDie('Time')
            total_time = 0
            for vehicle_id in range(data['num_vehicles']):
                while not routing.IsEnd(index):
                    time_var = time_dimension.CumulVar(index)
                    index = assignment.Value(routing.NextVar(index))
                time_var = time_dimension.CumulVar(index)
                total_time += assignment.Min(time_var)
                avg_length += total_time / (10 ** 6)
                count += 1
            print(total_time / (10**6))
            print('Total time of all routes: {}'.format(total_time / (10**6)))
    end = tm.time()
    print("avg length: ", avg_length / count)
    print("used time: ", end - start)


if __name__ == '__main__':
    sample_size = 10
    problem_size = 20
    dataset = generate_data(B=sample_size, size=problem_size)
    main(sample_size=sample_size)

