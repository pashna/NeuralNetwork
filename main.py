from Engine.NeuralNetwork import NeuralNetwork
import numpy as np

from sklearn import datasets
from sklearn.metrics import log_loss, mean_absolute_error


#print np.asarray([[0,1],[2,3]]) * np.asarray([[1,1],[1,1]])

network = NeuralNetwork(hidden_layers_sizes=[5, 5, 5], learning_rate=.1, max_iter=10000, test_size=0.3)
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
Y = iris.target

Y = Y[:100]
X = X[:100]
print np.unique(Y)
network.fit(X, Y)
y_pred = network.predict(X)
print "abs = ", mean_absolute_error(Y, y_pred)


"""
network = NeuralNetwork(hidden_layers_sizes=[3, 5], learning_rate=1., max_iter=10000, test_size=0.2)
X = []
N = 2
for i in range(N):
    X.extend([[0,1],[1,0],[1,1],[0, 0]])

X = np.asarray(X)
y = np.asarray([1,0,2,3]*N)
#y = np.asarray([1,0,]*5000)

network.fit(X, y)
print network.predict(np.asarray([[0,1],[1,0],[1,1],[0, 0]]))
"""