from LossAbstract import LossAbstract
import numpy as np
from sklearn.metrics import mean_squared_error


class MSE(LossAbstract):

    def derivative(self, y_true, y_pred):
        return y_true-y_pred


    def v_func(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)
        #return np.sum((y_true-y_pred)*(y_true-y_pred))/len(y_pred)


    def v_func_prob(self, Y_true, Y_pred):
        return np.sum((Y_true - Y_pred)**2)/len(Y_true)