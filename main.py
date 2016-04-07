from Engine.NeuralNetwork import NeuralNetwork
import numpy as np

from sklearn import datasets
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error
from Engine.ActivationFunction.Sigmoid import Sigmoid
from Engine.ActivationFunction.Identical import Identical


#print np.asarray([[0,1],[2,3]]) * np.asarray([[1,1],[1,1]])

#"""

def iris_network():
    iris = datasets.load_iris()
    X = iris.data  # we only take the first two features.
    y = iris.target
    print np.unique(y)

    network = NeuralNetwork(hidden_layers_sizes=[50, 50],
                            #activation_func=[Sigmoid(), Sigmoid(), Sigmoid()],
                            learning_rate=1.,
                            max_iter=300,
                            max_loss=0.1,
                            regularization='l1')
    network.fit(X, y)
    y_pred = network.predict(X)
    print "abs = ", mean_absolute_error(y, y_pred)

    print y
    print y_pred


def boston_network():
    boston_data = datasets.load_boston()
    X = boston_data.data
    y = boston_data.target
    #print y

    network = NeuralNetwork(hidden_layers_sizes=[2],
                  is_clasification=False,
                  learning_rate=0.000001,
                  max_iter=10000,
                  max_loss=40)

    X = np.asarray([[1, 5],
                    [2, 10],
                    [25, 3],
                    [50, 20],
                    [17, 12],
                    [44, 12],
                    [25, 11],
                    [77, 100],
                    [42,12],
                    [123,14],
                    [123,312],
                    [165,24],
                    [12, 42],
                    [123,124],
                    [123,50],
                    [12, 45],
                    [1, 5],
                    [2, 10],
                    [25, 3],
                    [50, 20],
                    [17, 12],
                    [44, 12],
                    [25, 11],
                    [77, 100],
                    [42,12],
                    [123,14],
                    [123,312],
                    [165,24],
                    [12, 42],
                    [123,124],
                    [123,50],
                    [12, 45]
                    ])

    y = X[:,0]*5 - X[:,1]*2 + 20

    #X = X[:,[0,1,2]]
    network.fit(X, y, test_size=0.15)
    y_pred = network.predict(X)
    print mean_squared_error(y, y_pred)
    print zip(y, y_pred)

if __name__ == "__main__":
    iris_network()
    #boston_network()


















"""
def simple():
    network = NeuralNetwork(hidden_layers_sizes=[2, 5], learning_rate=1., max_iter=10000, test_size=0.2)
    X = []
    N = 20
    for i in range(N):
        X.extend([[0,1],[1,0],[1,1],[0, 0]])

    X = np.asarray(X)
    y = np.asarray([1,0,2,3]*N)
    #y = np.asarray([1,0,]*5000)

    network.fit(X, y)
    print network.predict(np.asarray([[0,1],[1,0],[1,1],[0, 0]]))
"""