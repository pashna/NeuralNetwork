from LossAbstract import LossAbstract

class MSE(LossAbstract):

    def derivative(self, y_true, y_pred):
        return y_true-y_pred

    def v_func(self, y_true, y_pred):
        return sum(0.5*(y_true-y_pred)*(y_true-y_pred))