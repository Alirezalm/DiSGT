import numpy as np
from sklearn.linear_model import LogisticRegression

from gradient_tracking import GradientTracking, ComputationNetwork, Agent, ObjectiveFunction
from graph import RingGraph
from utils import create_random_qp, create_random_logistic_regression_data


class RandomQuadraticObj(ObjectiveFunction):

    def __init__(self, dim: int):
        self.P, self.c, self.d = create_random_qp(dim)

    def get_obj_at(self, x: np.ndarray) -> float:
        return float(0.5 * x.T @ self.P @ x + self.c.T @ x + self.d)

    def get_grad_at(self, x: np.ndarray) -> np.ndarray:
        return self.P @ x + self.c


def test_gradient_tracking_qp():
    N = 4
    n = 3
    topology = RingGraph(N)
    agents = []
    for i in range(N):
        obj = RandomQuadraticObj(dim=n)

        agents.append(Agent(obj))

    network = ComputationNetwork(topology, agents)

    gt = GradientTracking(network)
    x0 = np.zeros([n, 1])
    f_dist, x = gt.run(x0)

    # centralized solution
    P = np.zeros([n, n])
    c = np.zeros([n, 1])
    d = 0
    for agent in agents:
        P += agent.obj.P
        c += agent.obj.c
        d += agent.obj.d

    x_opt = np.linalg.solve(P, -c).reshape(n, 1)
    f_cent = float(0.5 * x_opt.T @ P @ x_opt + c.T @ x_opt + d)

    assert np.isclose(f_cent, f_dist)


class RandomLogRegObjective(ObjectiveFunction):
    def __init__(self, m: int, n: int):
        self.X, self.y = create_random_logistic_regression_data(m, n)

    def get_obj_at(self, x: np.ndarray) -> float:
        n = x.shape[0]
        h = self.logistic(x)
        return float(-self.y.T @ np.log(h) - (1 - self.y).T @ np.log(1 - h))

    def logistic(self, x: np.ndarray) -> np.ndarray:
        z = self.X @ x
        h = 1 / (1 + np.exp(-z))
        h[h == 1] = 1 - 1e-8
        h[h == 0] = 1e-8
        return h

    def get_grad_at(self, x: np.ndarray) -> np.ndarray:
        h = self.logistic(x)
        return self.X.T @ (h - self.y)


def test_gt_logistic_regression():
    N = 4
    n = 20
    m = 1000
    topology = RingGraph(N)
    agents = []
    for i in range(N):
        obj = RandomLogRegObjective(m, n)

        agents.append(Agent(obj))
    network = ComputationNetwork(topology, agents)

    gt = GradientTracking(network)
    x0 = np.zeros([n, 1])
    f_dist, x = gt.run(x0)

    # centralized solution
    X = agents[0].obj.X
    y = agents[0].obj.y
    for agent in agents[1:]:
        X = np.concatenate((X, agent.obj.X), axis=0)
        y = np.concatenate((y, agent.obj.y), axis=0)

    lr = LogisticRegression(penalty=None, fit_intercept=False)
    estimator = lr.fit(X, y.reshape(-1,))

    assert np.allclose(estimator.coef_.reshape(n, 1), x, rtol=1e-3)
