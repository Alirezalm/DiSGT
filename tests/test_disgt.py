import numpy as np

from disgt import SparsityEnforcer, DiSGT
from graph import RingGraph
from problems import SparseConvexQP, SparseLogisticRegression
from utils import get_sparsity, create_random_qp, create_random_logistic_regression_data


def test_sparsity_enforcer():
    sp_ef = SparsityEnforcer()
    n = 50
    x = np.random.randn(n, 1)
    lmbd = np.random.randn(n, 1)

    rho = 1
    M = 5
    kappa = 2
    y = sp_ef.enforce(x, lmbd, rho, M, kappa)
    assert get_sparsity(y) >= n - kappa


def test_check_nonzero():
    desired = 3
    n = 10
    x = np.zeros([n, 1])
    x[2] = 0.00001
    x[3] = 0.00001
    x[7] = 0.00001

    assert get_sparsity(x) >= desired


def test_disgt_for_qp():
    N = 10
    network = RingGraph(N)
    n = 10
    problems = []
    kappa = 5

    for i in range(N):
        Q, q, d = create_random_qp(n)
        problems.append(SparseConvexQP(Q, q, d, kappa))

    dg = DiSGT()
    dg.set_local_problems(problems)
    dg.set_network_topology(network)

    res = dg.optimize()

    assert get_sparsity(res.current_solution) >= n - kappa


def test_disgt_for_logreg():
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

    res = dg.optimize()
    assert get_sparsity(res.current_solution) >= n - kappa
