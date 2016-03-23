from ActivationAbs import ActivationAbs
from math import exp

class Sigmoid(ActivationAbs):

    def func(self, x):
        return 1.0/(1.0+exp(-x))

    def derivative(self, x):
        return self.func(x)*(1-self.func(x))