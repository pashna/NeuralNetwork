from LossAbstract import LossAbstract
import numpy as np

class MSE(LossAbstract):

    def derivative(self, y_true, y_pred):
        return y_true-y_pred


    def v_func(self, y_true, y_pred):
        return sum(0.5*(y_true-y_pred)*(y_true-y_pred))


    def v_func_prob(self, Y_true, Y_pred):
        return np.sum((Y_true - Y_pred)**2)/len(Y_true)