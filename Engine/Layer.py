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
        else:
            self._activation = activation


    def _init_weigth(self, n_enter, n_neural):
        """
        Инициализируем веса случайно [0, 0.5]
        :param n_enter:
        :param n_neural:
        """
        self._n_enter = n_enter
        self._n_neural = n_neural
        self._w = 0.1 * np.random.randn(n_neural, n_enter) + 0 #np.random.random((n_neural, n_enter))/10.
        #self._w = np.random.random((n_neural, n_enter))

        self._new_weight = self._w.copy()


    def propagate(self, x):
        x_ = np.append(x, 1) # insert - BIAS
        self.out = np.dot(self._w, x_.T)
        self.out = self._activation.v_func(self.out)

        return self.out


    def get_etalon_vector(self, y_label):
        """
        Возвращает эталонный вектор для задачи классификации.
        Например, есть три класса [0,1,2] и если y_label=2, то функция вернет
        [0,0,1]
        :param y_label:
        :return:
        """
        y = np.zeros(len(self.out))
        correct_index = self._label.index(y_label)
        y[correct_index] = 1
        return y


    def output_derivative(self):
        return self._activation.v_derivative(self.out)


    def get_class(self, y_output):
        """
        Функция возвращает значение класса по массиву output
        Имеет смысл только для задачи классификации
        :param y_output: массив из того
        :return:
        """
        max_index = np.argmax(y_output)
        return self._label[max_index]

    def update_weight(self, ):
        self._w = self._new_weight.copy()