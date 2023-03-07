from abc import ABC, abstractmethod

import numpy as np

from utils import is_convex


class ISparseProblem(ABC):
    @abstractmethod
    def compute_obj_at(self, x: np.ndarray) -> float:
        pass

    @abstractmethod
    def compute_grad_at(self, x: np.ndarray) -> np.ndarray:
        pass


class SparseConvexQP(ISparseProblem):

    def __init__(self, P: np.ndarray, c: np.ndarray, d: float):
        assert is_convex(P), "problem is not convex"

    def compute_obj_at(self, x: np.ndarray) -> float:
        pass

    def compute_grad_at(self, x: np.ndarray) -> np.ndarray:
        pass
