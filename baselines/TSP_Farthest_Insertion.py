# reference: https://github.com/wouterkool/attention-learn-to-route

import numpy as np
import time
from scipy.spatial import distance_matrix


def _calc_insert_cost(D, prv, nxt, ins):
    return (
            D[prv, ins]
            + D[ins, nxt]
            - D[prv, nxt]
    )


def run_insertion(loc):
    n = len(loc)
    D = distance_matrix(loc, loc)

    mask = np.zeros(n, dtype=bool)
    tour = []
    for i in range(n):
        feas = mask == 0
        feas_ind = np.flatnonzero(mask == 0)
        if i == 0:
            a = D.max(1).argmax()  # Node with farthest distance to any other node
        else:
            a = feas_ind[D[np.ix_(feas, ~feas)].min(1).argmax()]  # node which has closest node in tour farthest
        mask[a] = True

        if len(tour) == 0:
            tour = [a]
        else:
            ind_insert = np.argmin(
                _calc_insert_cost(
                    D,
                    tour,
                    np.roll(tour, -1),
                    a
                )
            )
            tour.insert(ind_insert + 1, a)

    cost = D[tour, np.roll(tour, -1)].sum()
    return cost, tour


def solve_insertion(loc):
    start = time.time()
    cost, tour = run_insertion(loc)
    duration = time.time() - start
    return cost, tour, duration


if __name__ == '__main__':
    # generate data
    dataset_size = 100
    tsp_size = 20
    dataset = np.random.uniform(size=(dataset_size, tsp_size, 2)).tolist()

    tot_cost, tot_duration = 0, 0
    for i in range(dataset_size):
        graph = dataset[i][:][:]
        cost, tour, duration = solve_insertion(graph)
        tot_cost += cost
        tot_duration += duration

    print("Average cost: ", tot_cost / dataset_size)
    print("Average duration: ", tot_duration / dataset_size)