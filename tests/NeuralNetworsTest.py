# -*- coding: utf-8 -*-
from Engine.NeuralNetwork import NeuralNetwork
import unittest
import numpy as np

class NetworkTest(unittest.TestCase):

    def test_forward_prop(self):
        network = NeuralNetwork(hidden_layers_sizes=[4, 1])
        X = np.asarray([[2, 4], [3, 6], [5, 10]])
        network.fit(X, np.asarray([1,2,3]))
        prop = network.forward_prop(X[0])
