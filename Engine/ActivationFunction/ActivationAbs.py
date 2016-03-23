# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import numpy as np

class ActivationAbs():
    __metaclass__ = ABCMeta

    def __init__(self):
        self.vectorize_function()


    def vectorize_function(self):
        self.v_func = np.vectorize(self.func)
        self.v_derivative = np.vectorize(self.derivative)


    @abstractmethod
    def func(self, x):
        """
        Возвращает значение функции активации
        """
        pass

    @abstractmethod
    def derivative(self, x):
        """
        Возвращает значение производной функции активации
        """
        pass