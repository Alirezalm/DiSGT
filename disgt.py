from dataclasses import dataclass
from typing import List, Optional

from problems import ISparseProblem, SparseConvexQP
from utils import is_pd


@dataclass
class AlgorithmSettings:
    max_iter: int = 1000
    gamma: float = 1e-2
    eps: float = 1e-6


class Agent:
    def __init__(self, problem: ISparseProblem):
        self.problem = problem


class Environment:

    # singleton environment class
    def __init__(self):
        self.agents: List[Agent] = []
        self.settings: AlgorithmSettings = AlgorithmSettings()  # loading default settings
        self.iteration: Iteration = Optional[None]
        self.result: Result = Optional[None]

    def register_agent(self, agent: Agent):
        self.agents.append(agent)

    def set_settings(self, settings: AlgorithmSettings):
        self.settings = settings


class Result:
    def __init__(self, env: Environment):
        self.env = env

        self.obj_values: List[float] = []
        self.iteration_number: int = 0


class Report:
    def __init__(self, env: Environment):
        self.env = env


class Iteration:
    def __init__(self, env: Environment):
        self.env = env
        self.env.result.iteration_number += 1


class DiSGT:
    """
    Implementation of DiSGT without MPI. The algorithm accepts a list of local problems subject to
    sparsity constraint. In this implementation, all problem types are the same at the moment.
    """

    def __init__(self):
        self.env = Environment()
        self.env.result = Result(self.env)
        self.env.iteration = Iteration(self.env)

        self.N = 0

    def optimize(self):
        pass

    def set_agents(self, agents: List[Agent]):
        self.validate(agents)

        for agent in agents:
            self.env.register_agent(agent)

        self.N = len(agents)

    @staticmethod
    def validate(agents: List[Agent]):

        assert len(agents) > 0, "number of agents must be > 0"
        for agent in agents:
            if isinstance(agent.problem, SparseConvexQP):
                assert is_pd(agent.problem.P), "problem is not convex"
