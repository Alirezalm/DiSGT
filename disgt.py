from dataclasses import dataclass
from typing import List

import numpy as np

from gradient_tracking import ObjectiveFunction, Agent, ComputationNetwork, GradientTracking
from graph import IGraph
from problems import ISparseProblem


class AugLagObjective(ObjectiveFunction):

    def __init__(self, p: ISparseProblem, lmbd: np.ndarray, y: np.ndarray, rho, N: int):
        self.p = p
        self.lmbd = lmbd
        self.y = y
        self.rho = rho
        self.N = N

    def get_obj_at(self, x: np.ndarray) -> float:
        return float(self.p.compute_obj_at(x) + (1 / self.N) * (
                (self.lmbd.T @ (x - self.y)) + self.rho / 2 * (np.linalg.norm(x - self.y, 2) ** 2)
        ))

    def get_grad_at(self, x: np.ndarray) -> np.ndarray:
        return self.p.compute_grad_at(x) + (1 / self.N) * (self.lmbd + self.rho * (x - self.y))


class PrimalSolver:
    def __init__(self, problems: List[ISparseProblem], graph: IGraph):
        self.problems = problems
        self.graph = graph

    def update_x(self, dual_vars: np.ndarray, y: np.ndarray, rho: float) -> np.ndarray:
        n = y.shape[0]
        agents = []
        N = len(self.problems)
        for problem in self.problems:
            obj = AugLagObjective(problem, dual_vars, y, rho, N)
            agents.append(Agent(obj))

        network = ComputationNetwork(self.graph, agents)

        gt = GradientTracking(network)

        x0 = np.zeros([n, 1])

        f_dist, x = gt.run(x0)

        return x


class SparsityEnforcer:
    pass


@dataclass
class DiSGTSettings:
    max_iter: int = 1000
    eps: float = 1e-6
    rho: float = 1


class DiSGT:
    def __init__(self, p: List[ISparseProblem], net_model: IGraph, s: DiSGTSettings = DiSGTSettings()):
        self.p = p
        self.s = s
        self.graph = net_model
        self.primal_solver = PrimalSolver(p, net_model)
        self.sparsity_enforcer = SparsityEnforcer()

    def optimize(self):
        err = 1e10
        n = self.p[0].get_dim()

        lmbd = np.zeros([n, 1])
        y = np.zeros([n, 1])
        rho = self.s.rho

        while err > self.s.eps:
            self.primal_solver.update_x(lmbd, y, rho)

