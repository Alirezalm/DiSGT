from abc import ABC
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from gradient_tracking import ObjectiveFunction, Agent, ComputationNetwork, GradientTracking
from graph import IGraph
from problems import ISparseProblem
import gurobipy as gp
from gurobipy import GRB

from tasks import TaskStartIteration, TaskManager, TaskAddPrimalSolution, TaskEnforceSparsity, TaskUpdateDualVariables
from utils import get_sparsity


class Environment:

    def __init__(self):
        self.primal_solver: "PrimalSolver" = Optional[None]

        self.sparsity_enforcer: "SparsityEnforcer" = Optional[None]

        self.iteration: "Iteration" = Optional[None]

        self.problems: "List[ISparseProblem]" = Optional[None]

        self.report: "Report" = Optional[None]

        self.results: "Result" = Optional[None]

        self.settings: "DiSGTSettings" = Optional[None]

        self.timer: "Timer" = Optional[None]

        self.network: "IGraph" = Optional[None]

        self.task_manager: "TaskManager" = Optional[None]


class Result:
    def __init__(self, env: Environment):
        self.env = env
        self.total_obj_val = 0
        self.total_obj_vals = []

        self.current_solution: np.ndarray = Optional[None]
        self.current_solutions: List[np.ndarray] = []

        self.current_aug_solution: np.ndarray = Optional[None]
        self.current_aug_solutions: List[np.ndarray] = []

        self.current_dual_solution: np.ndarray = Optional[None]
        self.current_dual_solutions: List[np.ndarray] = []

        self.iterations: List[Iteration] = []

        self.error = 1e10

    def set_total_obj_val(self, obj_val: float):
        self.total_obj_val = obj_val
        self.total_obj_vals.append(obj_val)

    def add_solution(self, x: np.ndarray):
        self.current_solution = x
        self.current_solutions.append(x)

    def add_aug_solution(self, y: np.ndarray):
        self.current_aug_solution = y
        self.current_aug_solutions.append(y)

    def add_dual_solution(self, lmbd: np.ndarray):
        self.current_dual_solution = lmbd
        self.current_dual_solutions.append(lmbd)

    def compute_error(self):
        return float(np.linalg.norm(self.current_solution - self.current_solution, 2))

    def get_current_iteration(self):
        return self.env.iteration

    def make_iteration(self):
        self.iterations.append(Iteration(self.env))

    def get_num_iter(self):
        return len(self.iterations)


class Iteration:
    def __init__(self, env: Environment):
        self.env = env
        self.iter_number = env.results.get_num_iter() + 1
        # stores interation information


class Report:
    pass


class Timer:
    pass


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
    def __init__(self, env: Environment):
        self.env = env

    def update_x(self, dual_vars: np.ndarray, y: np.ndarray, rho: float) -> np.ndarray:
        n = y.shape[0]
        agents = []
        N = len(self.env.problems)
        for problem in self.env.problems:
            obj = AugLagObjective(problem, dual_vars, y, rho, N)

            agents.append(Agent(obj))

        network = ComputationNetwork(self.env.network, agents)

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
    def __init__(self):
        self.env: Environment = Environment()

        # assert self.env.network, "network topology is not available"
        # assert len(self.env.problems) > 1, "number of problems must be greater than 1"
        self.env.results = Result(self.env)

        self.env.sparsity_enforcer = SparsityEnforcer()

        self.env.primal_solver = PrimalSolver(self.env)

        self.env.iteration = Iteration(self.env)

        self.env.settings = DiSGTSettings()

        self.env.report = Report()

        self.env.task_manager = TaskManager(self.env)

        self.initialize_task_queue()

    def set_local_problems(self, problems: List[ISparseProblem]):
        assert len(problems) > 1
        self.env.problems = problems

    def set_network_topology(self, network: IGraph):
        self.env.network = network

    def initialize_task_queue(self):
        task_init_iter = TaskStartIteration(self.env)
        self.env.task_manager.add_task(task_init_iter, "t_init_iter")

        task_solve_primal = TaskAddPrimalSolution(self.env)
        self.env.task_manager.add_task(task_solve_primal, "t_solve_primal")

        task_enforce_sparsity = TaskEnforceSparsity(self.env)
        self.env.task_manager.add_task(task_enforce_sparsity, "t_enforce_primal")

        task_add_dual_sol = TaskUpdateDualVariables(self.env)
        self.env.task_manager.add_task(task_add_dual_sol, "t_add_dual")

    def optimize(self):

        while not self.env.task_manager.is_task_queue_empty():
            for task, task_id in self.env.task_manager.tasks:
                if task.is_active():
                    task.initialize()
                    task.execute()

# def display_info(self, x, err):
#     log = f"{get_sparsity(x)} of {x.shape[0] - self.kappa}| {err: 4.4f}"
#
#     print(log)

# def optimize(self):
#     err = 1e10
#     n = self.p[0].get_dim()
#
#     lmbd = np.zeros([n, 1])
#     y = np.zeros([n, 1])
#     rho = self.s.rho
#
#     while err > self.s.eps:
#         x = self.primal_solver.update_x(lmbd, y, rho)
#         y = self.sparsity_enforcer.enforce(x, lmbd, rho, self.s.M, self.kappa)
#
#         lmbd += rho * (x - y)
#         err = np.linalg.norm(x - y)
#         # print(np.linalg.norm(x - y))
#         self.display_info(x, err)
#         f = self.compute_total_objective(x)
#
#     return x, f
#
# def compute_total_objective(self, x: np.ndarray):
#     return sum([
#         problem.compute_obj_at(x) for problem in self.p
#     ])
