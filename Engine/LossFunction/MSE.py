from LossAbstract import LossAbstract

class MSE(LossAbstract):

    def derivative(self, y_true, y_pred):
        return y_true-y_pred

    def func(self, y_true, y_pred):
        return 0.5*(y_true-y_pred)*(y_true-y_pred)