import numpy as np

from graph import RingGraph


def test_ring_graph_base_case():
    n = 2
    ring_graph = RingGraph(n)

    adj = ring_graph.get_adj()
    true_adj = np.eye(n)

    assert np.array_equal(adj, true_adj)


def test_ring_graph_normal_case():
    n = 4
    ring_graph = RingGraph(n)

    adj = ring_graph.get_adj()
    # adj_list_3 = [[0, 1, 1],
    #               [1, 0, 1],
    #               [1, 1, 0]]
    # adj_list_5 = [[0, 1, 0, 0, 1],
    #               [1, 0, 1, 0, 0],
    #               [0, 1, 0, 1, 0],
    #               [0, 0, 1, 0, 1],
    #               [1, 0, 0, 1, 0]]
    adj_list_4 = [[0, 1, 0, 1],
                  [1, 0, 1, 0],
                  [0, 1, 0, 1],
                  [1, 0, 1, 0]]

    true_adj = np.array(adj_list_4, dtype=float)

    assert np.array_equal(adj, true_adj)


def test_metro_hasting():
    n = 100
    ring_graph = RingGraph(n)
    true_weights = metropolis_hastings(ring_graph.get_adj())

    weights = ring_graph.get_weights()
    assert np.array_equal(weights, true_weights)


def metropolis_hastings(adj: np.ndarray, link_type: str = 'undirected') -> np.ndarray:
    N = np.shape(adj)[0]
    degree = np.sum(adj, axis=0)
    W = np.zeros([N, N])
    for i in range(N):
        N_i = np.nonzero(adj[i, :])[0]  # Fixed Neighbors
        for j in N_i:
            W[i, j] = 1 / (1 + np.max([degree[i], degree[j]]))
        W[i, i] = 1 - np.sum(W[i, :])
    return W
