# -*- coding: utf-8 -*-
from Engine.Layer import Layer
import numpy as np
from LossFunction.MSE import MSE

class NeuralNetwork():

    def __init__(self, hidden_layers_sizes, is_clasification=True, loss=None, learning_rate=0.1):
        """
        :param hidden_layers_sizes: список, состоящий из количества нейронов в каждом слое. Например [4,6,2] - 3 внутренних слоя
        :param is_clasification: Если решается задача классификации
        :param loss: если функция потерь отличная от mse. Объект, реализующий абстрактный класс LossAbstract
        """
        self._hidden_layers_sizes = hidden_layers_sizes
        self._is_clasification = is_clasification
        if loss is None:
            self.loss = MSE()

        self._learning_rate = learning_rate


    def _init_layers(self, input_size, y):
        """
        Функция инициализирует слои в сетке.
        :param input_size: Список, в котором задается количество нейронов в каждом слое
        :param y: вектор значений функции
        """
        prev_exit = input_size
        self._layers = []
        for size in self._hidden_layers_sizes:
            l = Layer(n_enter=prev_exit, n_neural=size)
            self._layers.append(l)
            prev_exit = size

        self._init_output_layer(y, prev_exit)


    def _init_output_layer(self, y, prev_layer_size):

        if self._is_clasification:
            classes = list(np.unique(y))
            layer = Layer(n_enter=prev_layer_size, n_neural=len(classes), label=classes)
            # TODO: для двух классов один выход сделать может?
            self._layers.append(layer)
        else:
            # регрессия, пока пропустим
            pass


    def fit(self, X, y):
        self._init_layers(X.shape[1], y)
        for i in range(len(y)):
            x = X[i]
            y_pred = self.forward_prop(x)
            self.back_prop(x, y[i])


    def forward_prop(self, x):
        for l in self._layers:
            x = l.propagate(x)
        return x


    def back_prop(self, x, y):
        self._calculate_output_layer(y)


    def _calculate_hidden_layers(self):
        pass



    def _calculate_output_layer(self, y):
        """
        Функция производит пересчет весов для выходного слоя
        :param y: реальное значение y
        """
        output_layer = self._layers[-1]
        prev_layer = self._layers[-2]
        grad = self._calculate_grad(y, output_layer.out)

        output_layer.delta = output_layer.output_derivative() * grad
        #self._learning_rate *
        output_layer._w += np.outer(output_layer.delta, prev_layer.out)


    def _generate_etalon_vector(self, y_true_label):
        """
        Функция генерирует эталонный выходной вектор.
        Без этого нельзя взять градиент
        :param y_true_label: истинное значение
        """
        return self._layers[-1].get_etalon_vector(y_true_label)


    def _calculate_grad(self, y, y_pred):
        """
        :param y: истинной значение y. int
        :param y_pred: вектор предсказанных значений
        """
        y_true = self._generate_etalon_vector(y)
        return self.loss.v_derivative(y_true, y_pred)
