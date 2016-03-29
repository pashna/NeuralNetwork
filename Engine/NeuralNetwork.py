# -*- coding: utf-8 -*-
from Engine.Layer import Layer
import numpy as np
from LossFunction.MSE import MSE
from LossFunction.BernoulliLikelyhood import BernoulliLikelyhood as Bernoulli
from sklearn import cross_validation
from random import randint


class NeuralNetwork():

    def __init__(self, hidden_layers_sizes, is_clasification=True, loss=None, learning_rate=0.5, max_iter=100, test_size=0.3, max_loss=0.1, random_state=None):
        """
        :param hidden_layers_sizes: список, состоящий из количества нейронов в каждом слое. Например [4,6,2] - 3 внутренних слоя
        :param is_clasification: Если решается задача классификации
        :param loss: если функция потерь отличная от mse. Объект, реализующий абстрактный класс LossAbstract
        """
        # TODO: ДОБАВИТЬ check_params
        # TODO: ДОБАВИТЬ bias!
        # TODO: ДОБАВИТЬ функции активации
        # TODO: ДОБАВИТЬ ИНИЦИАЛИЗВЦИЮ ИЗ РАСПРЕДЕЛЕНИЯ ГАУССА

        self._hidden_layers_sizes = hidden_layers_sizes
        self._is_clasification = is_clasification
        self._max_iter = max_iter
        self._test_size = test_size
        self._max_loss = max_loss

        if random_state is None:
            self._random_state = randint(1, 1000)
        else:
            self._random_state = random_state

        if loss is None:
            # Если функция потерь не установлена, ставим MSE для классификации и Бернули для регрессии
            if self._is_clasification:
                self.loss = MSE()
            else:
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

        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=self._test_size, random_state=self._random_state)

        iter_n = 0
        while not self._is_stop_criterion(X_test, y_test, iter_n):

            for i in range(len(y_train)):
                x = X_train[i]
                self.forward_prop(x)
                self.back_prop(x, y_train[i])

            iter_n += 1



    def forward_prop(self, x):
        for l in self._layers:
            x = l.propagate(x)
        return x


    def back_prop(self, x, y):
        self._calculate_output_layer(y)
        self._calculate_hidden_layers(x, y)
        self._update_weight()


    def _calculate_hidden_layers(self, x, y):

        """
        Пересчет весов для внутренних слоев
        :param x:
        :param y:
        """
        for i in range(1, len(self._layers)):
            next_layer = self._layers[-i]
            cur_layer = self._layers[-i-1]

            # Либо выход внутреннего слоя, либо входные x
            if abs(-i-2) > len(self._layers):
                in_value = x
            else:
                in_value = self._layers[-i-2].out


            #affs = next_layer._w[:,range(0, len(next_layer._w)-1)]

            delta = (np.dot(next_layer.delta, next_layer._w)).T
            delta = delta[:-1]
            der = cur_layer.output_derivative()
            delta *= cur_layer.output_derivative()
            cur_layer.delta = delta

            cur_layer._new_weight += self._learning_rate * np.outer(cur_layer.delta, np.append(in_value, 1))


    def _update_weight(self):
        """
        Функция обновляет веса во всех слоях на только что высчитанные
        """
        for layer in self._layers:
            layer.update_weight()


    def _calculate_output_layer(self, y):
        """
        Функция производит пересчет весов для выходного слоя
        :param y: реальное значение y
        """
        output_layer = self._layers[-1]
        prev_layer = self._layers[-2]
        grad = self._calculate_grad(y, output_layer.out)

        o = output_layer.output_derivative()
        output_layer.delta = output_layer.output_derivative() * grad

        #output_layer._new_weight += self._learning_rate * np.outer(output_layer.delta, prev_layer.out)
        #WEIGHT = self._learning_rate * np.outer(output_layer.delta, prev_layer.out) #append - BIAS
        output_layer._new_weight += self._learning_rate * np.outer(output_layer.delta, np.append(prev_layer.out, 1))


    def _generate_etalon_vector(self, y_true_label):
        """
        Функция генерирует эталонный выходной вектор.
        Без этого нельзя взять градиент
        :param y_true_label: истинное значение
        """
        return self._layers[-1].get_etalon_vector(y_true_label)


    def predict(self, X):
        result = []
        for x in X:
            y = self.forward_prop(x)
            result.append(self._layers[-1].get_value(y))

        return np.asarray(result)


    def _calculate_grad(self, y, y_pred):
        """
        :param y: истинной значение y. int
        :param y_pred: вектор предсказанных значений
        """
        y_true = self._generate_etalon_vector(y)
        return self.loss.v_derivative(y_true, y_pred)


    def _is_stop_criterion(self, X_test, y_test, iter_n):
        """
        Функция решает, не пора ли остановиться
        :param X:
        :param y:
        :param iter_n: номер итерации
        """
        y_predicted = self.predict(X_test)
        loss_value = self.loss.v_func(y_test, y_predicted)
        print loss_value,
        if (iter_n > self._max_iter) \
            or (self._max_loss > loss_value):
            return True
        return False


    def predict_prob(self, X):
        result = []
        for x in X:
            y = self.forward_prop(x)
            result.append(y/sum(y))

        return result
