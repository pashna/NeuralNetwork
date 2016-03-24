from Engine.NeuralNetwork import NeuralNetwork
import numpy as np


network = NeuralNetwork(hidden_layers_sizes=[4])
X = np.asarray([[2, 1], [500, 100], [2, 1], [2, 1], [2, 1], [2, 1], [2, 1], [2, 1], [2, 1], [2, 1], [2, 1], [2, 1], [2, 1], [2, 1], [2, 1], [2, 1]])
network.fit(X, np.asarray([1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
print network.forward_prop(np.asarray([2, 1]))