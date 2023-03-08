from disgt import DiSGT
from graph import RingGraph
from problems import SparseConvexQP
from utils import create_random_qp

N = 4
network = RingGraph(N)
n = 10
problems = []
kappa = 5

for i in range(N):
    Q, q, d = create_random_qp(n)
    problems.append(SparseConvexQP(Q, q, d, kappa))


dg = DiSGT(problems, network)
dg.optimize()

