# -*- coding: utf-8 -*-
from Engine.Layer import Layer
import numpy as np
from LossFunction.MSE import MSE
from ActivationFunction.Identical import Identical
from LossFunction.BernoulliLikelyhood import BernoulliLikelyhood as Bernoulli
from sklearn import cross_validation
from random import randint
import sys
import copy

class NeuralNetwork():

    def __init__(self, hidden_layers_sizes, activation_func=None, is_clasification=True, loss=None, learning_rate=0.5, max_iter=100, max_loss=0.1, classes=None):
        """
        :param hidden_layers_sizes: список, состоящий из количества нейронов в каждом слое. Например [4,6,2] - 3 внутренних слоя
        :param activation_func: список, содержащий объекты-функции активации. Реализуют абстрактный класс ActivationAbs
        :param is_clasification: Если решается задача классификации
        :param loss: если функция потерь отличная от mse. Объект, реализующий абстрактный класс LossAbstract
        :param learning_rate: float, изначальный learning_rate
        :param max_iter: int, максимальное количество итераций (вех)
        :param max_loss: минимальное значение функции потери, при котором продолжаем обучаться
        """
        self._hidden_layers_sizes = hidden_layers_sizes
        self.classes = classes

        self._activation_func = activation_func
        if activation_func is not None and is_clasification:
            if len(activation_func) != len(hidden_layers_sizes)+1 :
                print "Count of activation function should be correspond with count of layers. \nRule for classification: len(activation) == len(hidden_layers)+1"
                return
        if activation_func is not None and not is_clasification:
            if len(activation_func) != len(hidden_layers_sizes):
                print "Count of activation function should be correspond with count of layers. \nRule for regression: len(activation) == len(hidden_layers"
                return

        self._is_clasification = is_clasification
        self._max_iter = max_iter
        self._max_loss = max_loss
        self._learning_rate = learning_rate
        self.loss = loss

        if loss is None:
            # Если функция потерь не установлена, ставим MSE для классификации и Бернули для регрессии
            if self._is_clasification:
                self.loss = MSE()
            else:
                self.loss = MSE()

        # Поля для обновления learning_rate
        self._min_loss = 9999999.9
        self._incorrect_iter_n = 0


    def _init_layers(self, input_size, y):
        """
        Функция инициализирует слои в сетке.
        :param input_size: Список, в котором задается количество нейронов в каждом слое
        :param y: вектор значений функции
        """
        prev_exit = input_size
        self._layers = []

        for i in range(len(self._hidden_layers_sizes)):
            size = self._hidden_layers_sizes[i]

            if self._activation_func is not None:
                activation = self._activation_func[i]
            else:
                activation = None

            l = Layer(n_enter=prev_exit, n_neural=size, activation=activation)

            self._layers.append(l)
            prev_exit = size

        self._init_output_layer(y, prev_exit)


    def _init_output_layer(self, y, prev_layer_size):

        if self._is_clasification:
            #classes = list(np.unique(y))
            if self._activation_func is not None:
                activation = self._activation_func[-1]
            else:
                activation = None

            layer = Layer(n_enter=prev_layer_size, n_neural=len(self.classes), label=self.classes, activation=activation)
            self._layers.append(layer)
        else:
            # регрессия, пока пропустим
            layer = Layer(n_enter=prev_layer_size, n_neural=1, activation=Identical())
            self._layers.append(layer)


    def _cross_validate(self, X, y, test_size=0.25, random_state=None):
        """
        Функция выполняет разбиение X и y на тестовую и тренировочную
        :param X:
        :param y:
        :param test_size:
        :param random_state:
        :return:
        """
        if random_state is None:
            random_state = randint(1, 1000)
        else:
            random_state = random_state

        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test



    def fit(self, X, y, cv=None, test_size=0.25, random_state=None):
        """
        :param X: тренировочный Х
        :param y: тренировочный y
        :param cv: touple(validate_X, validate_y). Чтобы задействовать внутреннюю кросвалидацию оставить параметр None
        :param test_size: размер валидационной выборки при разделении X и y
        :param random_state: параметр для генератора случайных чисел
        """

        if cv is not None:
            # Если передано явно разбиение на валидационную и тренировочную выборку
            X_train = X
            y_train = y
            X_test = cv[0]
            y_test = cv[1]
        else:
            # Если разбиение явно не задано - кросвалидируемся
            X_train, X_test, y_train, y_test = self._cross_validate(X, y, test_size, random_state)

        self._init_layers(X.shape[1], y)
        iter_n = 0

        while not self._is_stop_criterion(X_test, y_test, iter_n):

            for i in range(len(y_train)):
                x = X_train[i]
                y = self.forward_prop(x)
                self.back_prop(x, y_train[i])
                self._update_weight()

            iter_n += 1


    def forward_prop(self, x):
        for l in self._layers:
            x = l.propagate(x)
        return x


    def back_prop(self, x, y):
        self._calculate_output_layer(y)
        self._calculate_hidden_layers(x, y)


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

            delta = (np.dot(next_layer.delta, next_layer._w)).T
            delta = delta[:-1]
            dev = cur_layer.output_derivative()
            delta *= cur_layer.output_derivative()
            cur_layer.delta = delta

            cur_layer._new_weight += self._learning_rate * np.outer(cur_layer.delta, np.append(in_value, 1))


    def _update_weight(self):
        """
        Функция обновляет веса во всех слоях на только что высчитанные
        """
        for layer in self._layers:
            layer.update_weight()


    def _calculate_output_layer(self, y_true):
        """
        Функция производит пересчет весов для выходного слоя
        :param y_true: float, реальное значение y
        """
        output_layer = self._layers[-1]
        prev_layer = self._layers[-2]
        grad = self._calculate_grad(y_true, output_layer.out)

        o = output_layer.output_derivative()
        output_layer.delta = output_layer.output_derivative() * grad
        output_layer._new_weight += self._learning_rate * np.outer(output_layer.delta, np.append(prev_layer.out, 1)) #append - BIAS


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
            if self._is_clasification:
                #list() ???
                result.append(self._layers[-1].get_class(y))
            else:
                result.append(y)

        return np.asarray(result)


    def _calculate_grad(self, y, y_pred):
        """
        :param y: истинной значение y. int
        :param y_pred: вектор предсказанных значений
        """
        if self._is_clasification:
            y_true = self._generate_etalon_vector(y)
        else:
            y_true = np.asarray([y])
        return self.loss.v_derivative(y_true, y_pred)


    def _is_stop_criterion(self, X_test, y_test, iter_n):
        """
        Функция решает, не пора ли остановиться
        :param X:
        :param y:
        :param iter_n: номер итерации
        """
        if self._is_clasification:
            y_predicted = self.predict_output(X_test)
            y_test_prob = self._generate_etalon_matrix(y_test)
            loss_value = self.loss.v_func_prob(y_test_prob, y_predicted)

        else:
            y_predicted = self.predict(X_test)
            y_predicted = y_predicted.ravel()
            loss_value = self.loss.v_func(y_test, y_predicted)

        if iter_n%20 == 1:
            print "iter = ", iter_n, "  loss=",loss_value, " min_loss=", self._min_loss

        self._update_learning_rate(loss_value, X_test, y_test)

        if (iter_n > self._max_iter) \
            or (self._max_loss > loss_value):
            return True
        return False


    def _generate_etalon_matrix(self, y_test):
        """
        Функция генерирует эталонную матрицу (для задачи классификации)
        Т.е. матрицу, с числом строк, равной количеству объектов, с числом столбцов, равным количеству классов
        Единичка стоит в столбце с верным классом - в остльных стоят нули
        :param y_test:
        :return:
        """
        result = []
        for y in y_test:
            result.append(self._generate_etalon_vector(y))

        return np.asarray(result)


    def predict_output(self, X):
        """
        Выдает output нейронной сетки
        :param X:
        :return:
        """
        result = []
        for x in X:
            y = self.forward_prop(x)
            result.append(y)#/sum(y))

        return result


    def _update_learning_rate(self, loss_value, X_test, y_test):
        """
        Функция выполняет уменьшение learning rate в соответвии с правилом от Павла на лекции
        :param loss_value: текущее значения функции потерть
        :param X_test:
        :param y_test:
        """
        # Если значение функции потери уменьшилось - фиксируем текущее состояние
        if self._min_loss > loss_value:
            self._min_loss = loss_value
            self._min_learning_rate = self._learning_rate
            self._best_layers = copy.deepcopy(self._layers)#self._layers.copy()
            self._incorrect_iter_n = 0

        # Считаем количество итераций, которые выходят за 10% от оптимального
        if loss_value > self._min_loss*1.1:
            self._incorrect_iter_n += 1
        else:
            self._incorrect_iter_n = 0

        # Если таких итераций больше 10, откатываемся на лучшее состояние и уменьшаем learning_rate
        if self._incorrect_iter_n > 10:
            self._layers = copy.deepcopy(self._best_layers)
            self._learning_rate /= 2.
            self._incorrect_iter_n = 0
            print "Roll Back! New learning_rate = ", self._learning_rate

            """
            if self._is_clasification:
                y_predicted = self.predict_output(X_test)
                y_test_prob = self._generate_etalon_matrix(y_test)
                loss_value = self.loss.v_func_prob(y_test_prob, y_predicted)
            else:
                y_predicted = self.predict_output(X_test)
                loss_value = self.loss.v_func(y_test, y_predicted)

            #print "new loss = ", loss_value
            """