from Engine.NeuralNetwork import NeuralNetwork
import numpy as np

from sklearn import datasets
from sklearn.metrics import mean_squared_error as mse

network = NeuralNetwork(hidden_layers_sizes=[2, 3], learning_rate=1., max_iter=10000)
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
Y = iris.target

Y = Y[:100]
X = X[:100]
print np.unique(Y)
network.fit(X, Y)


"""
network = NeuralNetwork(hidden_layers_sizes=[2, 10, 10], learning_rate=1., max_iter=10000)
X = []
N = 400
for i in range(N):
    X.extend([[0,1],[1,0],[1,1],[0, 0]])

X = np.asarray(X)
y = np.asarray([1,0,2,3]*N)
#y = np.asarray([1,0,]*5000)

network.fit(X[:500], y[:500])
print network.predict(np.asarray([[0,1],[1,0],[1,1],[0, 0]]))
"""