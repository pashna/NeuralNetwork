# -*- coding: utf-8 -*-

from ActivationAbs import ActivationAbs
from math import exp
from scipy.stats import logistic

from scipy.special import expit

class Sigmoid(ActivationAbs):

    def func(self, x):
        if abs(x) > 100:
            x /= abs(x)
            x *= 100
        return 1.0/(1.0+exp(-x))


    def derivative(self, out):
        """
        ПРОИЗВОДНАЯ, НО НЕ Х, а от OUT
        :param x:
        :return:
        """
        return out*(1-out)