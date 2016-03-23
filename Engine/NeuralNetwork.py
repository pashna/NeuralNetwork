# -*- coding: utf-8 -*-
from Engine.Layer import Layer


class NeuralNetwork():

    def __init__(self, hidden_layers_sizes):
        """
        :param hidden_layers_sizes: список, состоящий из количества нейронов в каждом слое. Например [4,6,2] - 3 внутренних слоя
        """
        self._hidden_layers_sizes = hidden_layers_sizes


    def _init_hidden_layers(self, input_size):
        prev_exit = input_size
        self._layers = []
        for size in self._hidden_layers_sizes:
            l = Layer(n_enter=prev_exit, n_neural=size)
            self._layers.append(l)
            prev_exit = size


    def fit(self, X, y):
        self._init_hidden_layers(X.shape[1])


    def forward_prop(self, x):
        for l in self._layers:
            x = l.propagate(x)

        return x