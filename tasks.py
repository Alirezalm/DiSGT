from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class TaskBase:

    def __init__(self, env: "Environment"):
        self.is_task_active = True

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def execute(self):
        pass

    def is_active(self):
        return self.is_task_active

    def activate(self):
        self.is_task_active = True

    def deactivate(self):
        self.is_task_active = False


class TaskManager:
    def __init__(self, env: "Environment"):
        self.env = env
        self.tasks: List[Tuple[TaskBase, str]] = []

    def add_task(self, task: TaskBase, task_id: str):
        self.tasks.append((task, task_id))

    def is_task_queue_empty(self):
        return len(self.tasks) < 1

    def clear_tasks(self):
        self.tasks = []

    def get_task(self, task_id: str):
        pass


class TaskAddPrimalSolution(TaskBase):

    def __init__(self, env: "Environment"):
        super().__init__(env)
        self.env = env
        self.current_lmbd = None
        self.current_aug_sol = None

    def initialize(self):
        if self.env.results.get_num_iter() == 1:
            n = self.env.problems[0].get_dim()
            self.env.results.add_dual_solution(np.zeros([n, 1]))
            self.env.results.add_aug_solution(np.zeros([n, 1]))
        self.current_lmbd = self.env.results.current_dual_solution
        self.current_aug_sol = self.env.results.current_aug_solution

    def execute(self):
        x = self.env.primal_solver.update_x(
            self.current_lmbd,
            self.current_aug_sol,
            self.env.settings.rho
        )

        self.env.results.add_solution(x)


class TaskEnforceSparsity(TaskBase):

    def __init__(self, env: "Environment"):
        super().__init__(env)
        self.env = env
        self.current_solution = None
        self.current_lmbd = None

    def initialize(self):
        self.current_lmbd = self.env.results.current_dual_solution
        self.current_solution = self.env.results.current_solution

    def execute(self):
        y = self.env.sparsity_enforcer.enforce(
            self.current_solution,
            self.current_lmbd,
            self.env.settings.rho,
            self.env.settings.M,
            self.env.problems[0].get_kappa()
        )

        self.env.results.add_aug_solution(y)


class TaskUpdateDualVariables(TaskBase):

    def __init__(self, env: "Environment"):
        super().__init__(env)
        self.env = env
        self.current_solution = None
        self.current_aug_solution = None
        self.current_dual_solution = None

    def initialize(self):
        self.current_solution = self.env.results.current_solution
        self.current_aug_solution = self.env.results.current_aug_solution
        self.current_dual_solution = self.env.results.current_dual_solution

    def execute(self):
        lmbd = self.current_dual_solution + self.env.settings.rho * (
                self.current_solution - self.current_aug_solution
        )

        self.env.results.add_dual_solution(lmbd)


class TaskStartIteration(TaskBase):

    def __init__(self, env: "Environment"):
        super().__init__(env)
        self.env = env

    def initialize(self):
        pass

    def execute(self):
        self.env.results.make_iteration()
