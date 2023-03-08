from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from utils import is_pd


class ISparseProblem(ABC):
    @abstractmethod
    def compute_obj_at(self, x: np.ndarray) -> float:
        pass

    @abstractmethod
    def compute_grad_at(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_dim(self):
        pass

    @abstractmethod
    def get_kappa(self):
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

    def get_dim(self):
        return self.n

    def get_kappa(self):
        return self.kappa


class SparseLogisticRegression(ISparseProblem):
    def get_dim(self):
        pass

    def compute_obj_at(self, x: np.ndarray) -> float:
        pass

    def compute_grad_at(self, x: np.ndarray) -> np.ndarray:
        pass


class SparseLinearRegression(ISparseProblem):
    def get_dim(self):
        pass

    def compute_obj_at(self, x: np.ndarray) -> float:
        pass

    def compute_grad_at(self, x: np.ndarray) -> np.ndarray:
        pass


@dataclass
class Node:
    obj: Callable[[np.ndarray], float] = Optional[None]
    grad: Callable[[np.ndarray], np.ndarray] = Optional[None]
    x: np.ndarray = None
    y: np.ndarray = None
