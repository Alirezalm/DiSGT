from gradient_tracking import Agent, ComputationNetwork, GradientTracking
from graph import RingGraph
from tests.test_gt import RandomLogRegObjective
import numpy as np
N = 10
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
