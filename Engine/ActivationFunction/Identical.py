# -*- coding: utf-8 -*-

from ActivationAbs import ActivationAbs
from math import exp
from scipy.stats import logistic

class Sigmoid(ActivationAbs):

    def func(self, x):
        return x


    def derivative(self, out):
        """
        ПРОИЗВОДНАЯ, НО НЕ Х, а от OUT
        :param x:
        :return:
        """
        return 1