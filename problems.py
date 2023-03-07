from abc import ABC, abstractmethod

import numpy as np

from utils import is_pd


class ISparseProblem(ABC):
    @abstractmethod
    def compute_obj_at(self, x: np.ndarray) -> float:
        pass

    @abstractmethod
    def compute_grad_at(self, x: np.ndarray) -> np.ndarray:
        pass


class SparseConvexQP(ISparseProblem):

    def __init__(self, P: np.ndarray, c: np.ndarray, d: float, kappa: int):
        assert is_pd(P), "problem is not convex"
        self.n = P.shape[0]
        self.P = P

        assert self.n == c.shape[0], "dimension mismatched"
        self.c = c

        assert kappa < self.n, "number of non-zeros cannot exceed dim"
        self.kappa = kappa
        self.d = d

    def compute_obj_at(self, x: np.ndarray) -> float:
        assert x.shape[0] == self.n, "dim mismatch"
        return float(
            0.5 * x.T @ self.P @ x + self.c.T @ x + self.d
        )

    def compute_grad_at(self, x: np.ndarray) -> np.ndarray:
        assert x.shape[0] == self.n, "dim mismatch"

        return self.P @ x + self.c


class SparseLogisticRegression(ISparseProblem):
    def compute_obj_at(self, x: np.ndarray) -> float:
        pass

    def compute_grad_at(self, x: np.ndarray) -> np.ndarray:
        pass


class SparseLinearRegression(ISparseProblem):
    def compute_obj_at(self, x: np.ndarray) -> float:
        pass

    def compute_grad_at(self, x: np.ndarray) -> np.ndarray:
        pass
