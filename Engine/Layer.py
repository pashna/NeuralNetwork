# coding: utf-8
import numpy as np
from ActivationFunction.Sigmoid import Sigmoid
from ActivationFunction.Identical import Identical

class Layer():

    def __init__(self, n_enter, n_neural, activation=None, label=None, is_output=False):
        """
        :param n_enter: Количество входов (количество выходов предыдущего слоя)
        :param n_neural: Количество нейронов в слое
        :param activation: функция активации. объект, реализующий абстрактный класс ActivationAbs
        :param label: label'ы для классов
        """
        self._init_weigth(n_enter+1, n_neural) # +1 - bias
        self._label = label
        self._is_output = is_output

        if activation is None:
            self._activation = Sigmoid()#Identical()


    def _init_weigth(self, n_enter, n_neural):
        """
        Инициализируем веса случайно [0, 0.5]
        :param n_enter:
        :param n_neural:
        """
        self._n_enter = n_enter
        self._n_neural = n_neural
        self._w = np.random.random((n_neural, n_enter))/2.5

        #self._w /= self._w

        self._new_weight = self._w.copy()


    def propagate(self, x):
        x_ = np.append(x, 1) # insert - BIAS
        self.out = np.dot(self._w, x_.T)
        self.out = self._activation.v_func(self.out)
        """
        if not self._is_output:
            self.out = np.append(self.out, 1)
        """
        """
        if not self._is_output:
            self.out = np.append(self.out, 1)
        """

        return self.out


    def get_etalon_vector(self, y_label):
        y = np.zeros(len(self.out))
        correct_index = self._label.index(y_label)
        y[correct_index] = 1
        return y


    def output_derivative(self):
        aa = self._activation.v_derivative(self.out)
        return self._activation.v_derivative(self.out)


    def get_value(self, y):
        max_index = np.argmax(y)
        return self._label[max_index]

    def update_weight(self, ):
        self._w = self._new_weight.copy()