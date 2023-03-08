from disgt import DiSGT
from graph import RingGraph
from problems import SparseLogisticRegression
from utils import create_random_logistic_regression_data

N = 10
network = RingGraph(N)
n = 10
m = 100
problems = []
kappa = 5

for i in range(N):
    X, y = create_random_logistic_regression_data(m, n)

    problems.append(SparseLogisticRegression(X, y, kappa))

dg = DiSGT()
dg.set_local_problems(problems)
dg.set_network_topology(network)

x, f = dg.optimize()
