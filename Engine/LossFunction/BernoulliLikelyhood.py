from LossAbstract import LossAbstract

from math import log
import numpy as np

class BernoulliLikelyhood(LossAbstract):

    def v_func_prob(self, Y_true, Y_pred):
        sum_vector = np.sum(Y_pred, axis=1)
        Y_pred /= sum_vector[:,None]
        return -np.sum(Y_true * np.log(Y_pred))/len(Y_true)


    def derivative(self, y_true, y_pred):
        return y_true/y_pred - (1-y_true)/(1-y_pred)

    def v_func(self, y_true, y_pred):
        return sum(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))