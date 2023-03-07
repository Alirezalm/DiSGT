from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from graph import IGraph
from problems import ISparseProblem, SparseConvexQP, Node
from utils import is_pd
import numpy as np


class IAlgorithm(ABC):

    @abstractmethod
    def run(self):
        pass


@dataclass
class GtSettings:
    eps: float = 1e-6
    max_iter: int = 1e3
    alpha: float = 1e-2


@dataclass
class AlgorithmSettings:
    max_iter: int = 1000
    rho: float = 1e-2
    eps: float = 1e-6


class Agent:
    def __init__(self, problem: ISparseProblem):
        self.problem = problem


class GradientTracking(IAlgorithm):
    def __init__(self, graph: IGraph, settings: GtSettings = GtSettings()):
        self.nodes: List[Node] = Optional[None]
        self.graph = graph
        self.s = settings
        self.adj = self.graph.get_adj()
        self.weights = self.graph.get_weights()

    def set_nodes(self, nodes: List[Node]):
        self.nodes = nodes

    def run(self):
        err = 1e3
        temp_x = []
        temp_y = []
        while err > self.s.eps:

            for node_index, problem in enumerate(self.nodes):
                neighbours = self.get_neighbour(self.adj, node_index)
                x = self.estimate_x(node_index, neighbours)
                y = self.estimate_gradient(node_index, neighbours, x)
                temp_x.append(x)
                temp_y.append(y)

            self.update_nodes_data(temp_x, temp_y)

    @staticmethod
    def get_neighbour(adj, node_index):
        return np.where(adj[node_index, :] != 0)[0]

    def estimate_x(self, in_ind: int, neighbours: List[int]):
        data = []
        for n in neighbours:
            data.append(self.nodes[n].x)

        aggr = self.aggregate(in_ind,
                              self.nodes[in_ind].x,
                              neighbours,
                              data,
                              self.weights)

        return aggr - self.s.alpha * self.nodes[in_ind].y

    def estimate_gradient(self, in_ind: int, neighbours: List[int], x_current: np.ndarray):
        data = []

        for n in neighbours:
            data.append(self.nodes[n].y)

        aggr = self.aggregate(in_ind,
                              self.nodes[in_ind].y,
                              neighbours,
                              data,
                              self.weights)

        return aggr + self.nodes[in_ind].grad(x_current) - self.nodes[in_ind].grad(self.nodes[in_ind].x)

    def aggregate(self, current_node_index: int,
                  current_node_data: np.ndarray,
                  neighbours: List[int],
                  data: List[np.ndarray],
                  weights: np.ndarray):

        N = len(self.nodes)
        assert len(data) == len(neighbours), "mismatched dim"

        agg_result = np.zeros([N, N])

        for neighbour in neighbours:
            agg_result += weights[current_node_index, neighbour] * data[neighbour]

        agg_result += weights[current_node_index, current_node_index] * current_node_data

        return agg_result

    def update_nodes_data(self, temp_x: List[np.ndarray], temp_y: List[np.ndarray]):
        """
        node sync update
        :param temp_x:
        :param temp_y:
        :return:
        """
        for ind, node in enumerate(self.nodes):
            node.x = temp_x[ind]
            node.y = temp_y[ind]


class Environment:

    # singleton environment class
    def __init__(self):
        self.agents: List[Agent] = []
        self.settings: AlgorithmSettings = AlgorithmSettings()  # loading default settings
        self.iteration: Iteration = Optional[None]
        self.result: Result = Optional[None]
        self.gt: GradientTracking = Optional[None]
        self.graph: IGraph = Optional[None]

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

    def set_network_topology(self, graph: IGraph):
        self.env.graph = graph

    @staticmethod
    def validate(agents: List[Agent]):

        assert len(agents) > 0, "number of agents must be > 0"
        for agent in agents:
            if isinstance(agent.problem, SparseConvexQP):
                assert is_pd(agent.problem.P), "problem is not convex"
