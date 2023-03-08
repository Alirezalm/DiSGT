from dataclasses import dataclass
from typing import List

import numpy as np

from gradient_tracking import ObjectiveFunction, Agent, ComputationNetwork, GradientTracking
from graph import IGraph
from problems import ISparseProblem

import gurobipy as gp
from gurobipy import GRB

from utils import get_sparsity


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
        gt.cfg.verbose = False

        x0 = np.zeros([n, 1])

        f_dist, x = gt.run(x0)

        return x


class SparsityEnforcer:
    def __init__(self):
        self.model = None

    def enforce(self, x: np.ndarray, lmbd: np.ndarray, rho: float, M: float, kappa: int):
        self.model = gp.Model(SparsityEnforcer.__name__)
        n = x.shape[0]
        y = self.model.addMVar(shape=(n, 1), lb=-GRB.INFINITY)
        delta = self.model.addMVar(shape=(n, 1), vtype=GRB.BINARY)

        objective = -lmbd.T @ y + (rho / 2) * (float(x.T @ x) - 2 * x.T @ y + y.T @ y)
        self.model.setObjective(objective)

        for i in range(n):
            self.model.addConstr(y[i] <= M * delta[i], name=f'b1{i}')
            self.model.addConstr(-M * delta[i] <= y[i], name=f'b2{i}')

        self.model.addConstr(delta.sum() <= kappa, name='d')
        self.model.setParam('OutputFlag', 0)
        self.model.optimize()

        return y.x.reshape(n, 1)


@dataclass
class DiSGTSettings:
    max_iter: int = 1000
    eps: float = 1e-6
    rho: float = 25
    M: float = 5


class DiSGT:
    def __init__(self, p: List[ISparseProblem], net_model: IGraph, s: DiSGTSettings = DiSGTSettings()):
        self.p = p
        self.s = s
        self.graph = net_model
        self.primal_solver = PrimalSolver(p, net_model)
        self.sparsity_enforcer = SparsityEnforcer()
        self.kappa = p[0].get_kappa()

    def display_info(self, x, err):
        log = f"{get_sparsity(x)} of {x.shape[0] - self.kappa}| {err: 4.4f}"

        print(log)

    def optimize(self):
        err = 1e10
        n = self.p[0].get_dim()

        lmbd = np.zeros([n, 1])
        y = np.zeros([n, 1])
        rho = self.s.rho

        while err > self.s.eps:
            x = self.primal_solver.update_x(lmbd, y, rho)
            y = self.sparsity_enforcer.enforce(x, lmbd, rho, self.s.M, self.kappa)

            lmbd += rho * (x - y)
            err = np.linalg.norm(x - y)
            # print(np.linalg.norm(x - y))
            self.display_info(x, err)
            f = self.compute_total_objective(x)

        return x, f

    def compute_total_objective(self, x: np.ndarray):
        return sum([
            problem.compute_obj_at(x) for problem in self.p
        ])
