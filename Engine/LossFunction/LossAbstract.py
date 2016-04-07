# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import numpy as np

class LossAbstract():
    __metaclass__ = ABCMeta

    def __init__(self):#, regularizator=None, r_coef=0.01):
        """
        self.regularizator = regularizator
        self.r_coef = r_coef
        """
        self.vectorize_function()


    def vectorize_function(self):
        """
        Функция векторизирует функцию активации. Всем оптимизации, поцаны!
        """
        self.v_derivative = np.vectorize(self.derivative)

    """
    def v_derivative(self, y_true, y_pred, W):
        if self.regularizator is not None:
            return self.v_derivative_loss(y_true, y_pred) + self.r_coef * self.regularizator.derivative_matrix(W)
        else:
            return self.v_derivative_loss(y_true, y_pred)
    """

    @abstractmethod
    def v_func(self, y_true, y_pred):
        """
        Возвращает значение функции потери
        y_true и y_pred - векторы
        """
        pass

    @abstractmethod
    def v_func_prob(self, Y_true, Y_pred):
        """
        Возвращает значение функции потери,
        y_true и y_pred - матрицы, (Количество_Объектов)х(Количество_Классов)
        """
        pass

    @abstractmethod
    def derivative(self, y_true, y_pred):
        """
        Возвращает значение производной функции потери
        y_true и y_pred - числа
        """
        pass