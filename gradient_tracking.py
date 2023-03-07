from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List

import numpy as np

from graph import IGraph


class ObjectiveFunction(ABC):
    @abstractmethod
    def get_obj_at(self, x: np.ndarray) -> float:
        pass

    @abstractmethod
    def get_grad_at(self, x: np.ndarray) -> np.ndarray:
        pass


@dataclass
class GtConfig:
    max_iter: int = 1e4
    gamma: float = 1e-2
    eps: float = 1e-6


class Agent:
    def __init__(self, obj: ObjectiveFunction):
        self.obj = obj
        self.current_x_estimate: np.ndarray = Optional[None]
        self.current_y_estimate: np.ndarray = Optional[None]


class ComputationNetwork:
    def __init__(self, graph: IGraph, agents: List[Agent]):
        self.graph = graph
        self.agents = agents

    def get_adjacency_matrix(self) -> np.ndarray:
        return self.graph.get_adj()

    def get_double_stochastic_weights(self) -> np.ndarray:
        return self.graph.get_weights()

    def get_neighbours_index_by_node_index(self, node_index: int):
        return self.graph.get_neighbours_of(node_index)

    def get_size(self):
        N = len(self.agents)
        assert N > 1, "too small network"
        return N


class TemporaryStorage:
    def __init__(self, network: ComputationNetwork):
        self.x_temp = []
        self.y_temp = []
        self.network = network

    def add_new_estimates(self, x: np.ndarray, y: np.ndarray):
        self.x_temp.append(x)
        self.y_temp.append(y)

    def update_nodes(self):
        assert len(self.x_temp) == self.network.get_size()
        for ind, agent in enumerate(self.network.agents):
            agent.current_x_estimate = self.x_temp[ind]
            agent.current_y_estimate = self.x_temp[ind]
        self.reset()

    def reset(self):
        self.x_temp = []
        self.y_temp = []


class GradientTracking:
    def __init__(self, network: ComputationNetwork, cfg: GtConfig = GtConfig()):
        self.network = network
        self.cfg = cfg
        self.N = self.network.get_size()
        self.n = None
        self.weights = self.network.get_double_stochastic_weights()
        self.tmp_storage = TemporaryStorage(network)

    def initialize(self):
        for agent in self.network.agents:
            agent.current_x_estimate = np.zeros([self.n, 1])
            agent.current_y_estimate = agent.obj.get_grad_at(agent.current_x_estimate)

    def run(self, x0: np.ndarray):
        self.n = x0.shape[0]
        err = 1e10
        self.initialize()
        f = self.compute_total_obj()
        while err > self.cfg.eps:

            for agent_index, agent in enumerate(self.network.agents):
                neighbours = self.network.get_neighbours_index_by_node_index(agent_index)
                x = self.estimate_x(agent_index, neighbours)
                y = self.estimate_y(agent_index, neighbours, x)
                self.tmp_storage.add_new_estimates(x, y)

            self.update_agents()
            fold = f
            f = self.compute_total_obj()
            err = abs(f - fold)

    def update_agents(self):
        self.tmp_storage.update_nodes()

    def estimate_x(self, node_index: int, in_index: List[int]) -> np.ndarray:

        aggr = np.zeros([self.n, 1])

        for k in in_index:
            aggr += self.weights[node_index, k] * self.network.agents[k].current_x_estimate

        aggr += self.weights[node_index, node_index] * self.network.agents[node_index].current_x_estimate

        return aggr - self.cfg.gamma * self.network.agents[node_index].current_y_estimate

    def estimate_y(self, node_index: int, in_index: List[int], new_estimate_x: np.ndarray):

        aggr = np.zeros([self.n, 1])

        for k in in_index:
            aggr += self.weights[node_index, k] * self.network.agents[k].current_y_estimate

        aggr += self.weights[node_index, node_index] * self.network.agents[node_index].current_y_estimate
        x_old = self.network.agents[node_index].current_x_estimate

        grad = self.network.agents[node_index].obj.get_grad_at(new_estimate_x)
        grad_old = self.network.agents[node_index].obj.get_grad_at(x_old)

        return aggr + grad - grad_old

    def compute_total_obj(self) -> float:
        return sum([agent.obj.get_obj_at(agent.current_x_estimate) for agent in self.network.agents])
