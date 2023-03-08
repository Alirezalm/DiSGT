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


def create_random_logistic_regression_data(m: int = 100, n: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    X = preprocessing.normalize(randn(m, n), norm="l2")
    y = randn(m, 1)
    y[y >= 0.5] = 1
    y[y < 0.5] = 0

    return X, y


def get_sparsity(x: np.ndarray) -> int:
    n_zero = 0
    for value in x:

        if np.isclose(value, 0.0, atol=1e-5):
            n_zero += 1

    return n_zero
