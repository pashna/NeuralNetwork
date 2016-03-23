# coding: utf-8
import numpy as np
from ActivationFunction.Sigmoid import Sigmoid

class Layer():

    def __init__(self, n_enter, n_neural, activation=None, label=None):
        """
        :param n_enter: Количество входов (количество выходов предыдущего слоя)
        :param n_neural: Количество нейронов в слое
        :param activation: функция активации. объект, реализующий абстрактный класс ActivationAbs
        :param label: label'ы для классов
        """
        self._init_weigth(n_enter, n_neural)
        self._label = label

        if activation is None:
            self._activation = Sigmoid()


    def _init_weigth(self, n_enter, n_neural):
        """
        Инициализируем веса случайно [0, 0.5]
        :param n_enter:
        :param n_neural:
        """
        self._n_enter = n_enter
        self._n_neural = n_neural
        self._w = np.random.random((n_neural, n_enter))/2


    def propagate(self, x):
        self.out = np.dot(self._w, x.T)
        self.out = self._activation.v_func(self.out)
        return self.out

    def delta(self, antigrad):
        self._delta = self._activation.derivative(self.out)


    def get_etalon_vector(self, y_label):
        y = np.zeros(len(self.out))
        correct_index = self._label.index(y_label)
        y[correct_index] = 1
        return y

    def output_derivative(self):
        return self._activation.v_derivative(self.out)