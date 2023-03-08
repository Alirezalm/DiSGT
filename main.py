# from gradient_tracking import ObjectiveFunction, Agent, ComputationNetwork, GradientTracking
# from graph import RingGraph
# from utils import create_random_qp
# import numpy as np
#
#
# class QuadraticObj(ObjectiveFunction):
#
#     def __init__(self, dim: int):
#         self.P, self.c, self.d = create_random_qp(dim)
#
#     def get_obj_at(self, x: np.ndarray) -> float:
#         return float(0.5 * x.T @ self.P @ x + self.c.T @ x + self.d)
#
#     def get_grad_at(self, x: np.ndarray) -> np.ndarray:
#         return self.P @ x + self.c
#
#
# if __name__ == '__main__':
#     N = 4
#     n = 10
#     topology = RingGraph(N)
#     agents = []
#     for i in range(N):
#         obj = QuadraticObj(dim=n)
#         agents.append(Agent(obj))
#
#     network = ComputationNetwork(topology, agents)
#
#     gt = GradientTracking(network)
#     x0 = np.zeros([n, 1])
#     obj_val = gt.run(x0)
