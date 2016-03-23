# coding: utf-8
import numpy as np
from ActivationFunction.Sigmoid import Sigmoid

class Layer():

    def __init__(self, n_enter, n_neural, activation=None):
        """
        :param n_enter: Количество входов (количество выходов предыдущего слоя)
        :param n_neural: Количество нейронов в слое
        :param activation: функция активации. объект, реализующий абстрактный класс ActivationAbs
        """
        self._init_weigth(n_enter, n_neural)
        if activation is None:
            self._activation = Sigmoid()



    def _init_weigth(self, n_enter, n_neural):
        """
        Инициализируем веса
        :param n_enter:
        :param n_neural:
        """
        self._w = np.random.random((n_enter, n_neural))/2

        self._w /= self._w


    def propagate(self, x):
        self.out = np.dot(x, self._w)
        self.out = self._activation.v_func(self.out)
        return self.out