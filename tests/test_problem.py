from problems import SparseConvexQP
from utils import is_pd, create_random_qp


def test_is_pd():
    n = 20
    Q, q, d = create_random_qp(n)
    assert is_pd(Q)


def test_qp_problem():
    n = 20
    Q, q, d = create_random_qp(n)
    sqp = SparseConvexQP(Q, q, d, 4)

    assert is_pd(sqp.P)
