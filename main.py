from Engine.NeuralNetwork import NeuralNetwork
import numpy as np


network = NeuralNetwork(hidden_layers_sizes=[5], learning_rate=.1)
X = []
for i in range(5000):
    X.extend([[0,1],[1,0],[1,1],[0, 0]])

X = np.asarray(X)
y = np.asarray([1,0,2,3]*5000)

network.fit(X[:4000], y[:4000])

print network.predict(np.asarray([[0,1],[1,0],[1,1],[0, 0]]))