# -*- coding: utf-8 -*-
from Engine.NeuralNetwork import NeuralNetwork
import unittest
import numpy as np

class NetworkTest(unittest.TestCase):

    def test_forward_prop(self):
        """
        network = NeuralNetwork(hidden_layers_sizes=[4, 1])
        X = np.asarray([[2, 1], [3, 0], [5, 1]])
        network.fit(X, np.asarray([1,1,0]))
        prop = network.forward_prop(X[0])
        g = 100
        """


    def test_calculate_output_layer(self):
        network = NeuralNetwork(hidden_layers_sizes=[4])
        [[2, 1], [400, 500], [2, 1], [2, 1]]
        X = np.asarray([[2, 1], [400, 500], [2, 1], [2, 1], [2, 1], [2, 1], [2, 1],[2, 1],[2, 1],[2, 1],[2, 1],[2, 1],[2, 1]])
        y = np.asarray([1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
        for i in range(3, len(X)):
            network.fit(X[:i], y[:i])
            print network.forward_prop(np.asarray([2,1]))


