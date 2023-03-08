import numpy as np

from disgt import SparsityEnforcer
from utils import get_sparsity


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
