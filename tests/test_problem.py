from utils import is_convex, create_random_qp


def test_is_convex():
    n = 20
    Q, q, d = create_random_qp(n)

    assert is_convex(Q)



