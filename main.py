import utils
from disgt import DiSGT
from graph import RingGraph
from problems import SparseLogisticRegression
from utils import create_random_logistic_regression_data


def sparse_logistic_regression():
    N = 3

    n = 50
    m = 1000
    problems = []
    kappa = 2

    for i in range(N):
        X, y = create_random_logistic_regression_data(m, n)

        problems.append(SparseLogisticRegression(X, y, kappa))

    network = RingGraph(N)
    dg = DiSGT()
    dg.set_local_problems(problems)
    dg.set_network_topology(network)

    res = dg.optimize()
    print(f"number of non-zeros in solution: {n - utils.get_sparsity(res.current_solution)}")

    print(f"objval: {res.total_obj_val}")


if __name__ == '__main__':
    sparse_logistic_regression()
