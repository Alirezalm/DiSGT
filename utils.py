from typing import Tuple

import numpy as np
from numpy.random import randn, rand
from sklearn import preprocessing


def is_pd(Q: np.ndarray):
    eig = np.linalg.eig(Q)[0]
    return eig.all() >= 0


def create_random_qp(n: int = 2) -> Tuple[np.ndarray, np.ndarray, float]:

    Q = preprocessing.normalize(randn(n, n), norm="l2")
    q = randn(n, 1)
    d = randn()

    Q = 0.5 * (Q + Q.T)
    V = np.linalg.eig(Q)[1]

    Q = V @ np.diagflat(1 + rand(n, 1)) @ V.T

    return Q, q, d
