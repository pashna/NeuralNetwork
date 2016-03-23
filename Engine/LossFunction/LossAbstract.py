# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import numpy as np

class LossAbstract():
    __metaclass__ = ABCMeta

    def __init__(self):
        self.vectorize_function()


    def vectorize_function(self):
        """
        Функция векторизирует функцию активации. Всем оптимизации, поцаны!
        """
        self.v_func = np.vectorize(self.func)
        self.v_derivative = np.vectorize(self.derivative)


    @abstractmethod
    def func(self, y_true, y_pred):
        """
        Возвращает значение функции потери
        """
        pass

    @abstractmethod
    def derivative(self, y_true, y_pred):
        """
        Возвращает значение производной функции потери
        """
        pass