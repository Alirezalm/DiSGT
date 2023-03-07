from abc import ABC, abstractmethod
from builtins import staticmethod

import numpy as np
from numpy import zeros


class IGraph(ABC):

    @abstractmethod
    def get_adj(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_weights(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_neighbours_of(self, node_index: int):
        pass


class RingGraph(IGraph):

    def __init__(self, nnodes: int) -> None:
        assert nnodes > 1, "number of nodes cannot be 1"
        self.nnodes = nnodes
        self.adj: np.ndarray = zeros([self.nnodes, self.nnodes])
        self.__gen()

    def __gen(self):
        if self.nnodes == 2:
            self.adj = np.eye(self.nnodes)
            return

        for row in range(self.nnodes):
            for col in range(self.nnodes):
                self.__create_ring_edge(row, col)

    def __create_ring_edge(self, i, j):
        if i == j:
            if j < self.nnodes - 1:
                self.adj[i, j - 1] = 1
                self.adj[i, j + 1] = 1
            else:
                self.adj[i, j - 1] = 1
                self.adj[i, 0] = 1

    def get_adj(self) -> np.ndarray:

        return self.adj

    def get_neighbours_of(self, node_index: int):
        return np.where(self.adj[node_index, :] != 0)[0]

    def __metro_hasting(self):
        weights: np.ndarray = zeros([self.nnodes, self.nnodes])
        for row in range(self.nnodes):
            for col in range(self.nnodes):
                if row != col and self.adj[row, col] == 1:
                    di = np.sum(self.adj[row, :])
                    dj = np.sum(self.adj[col, :])
                    weights[row, col] = 1 / (max(di, dj) + 1)
                elif col == row:
                    di = np.sum(self.adj[row, :])
                    neighbours = np.where(self.adj[col, :] != 0)[0]
                    aggr = 0
                    for neighbour in neighbours:
                        dk = sum(self.adj[neighbour, :])
                        aggr += (1 / (max(di, dk) + 1))
                    weights[row, col] = 1 - aggr
        return weights

    def get_weights(self) -> np.ndarray:
        return self.__metro_hasting()
