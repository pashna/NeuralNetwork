import numpy as np


class L1():

    def derivative_matrix(self, W):
        return np.sign(W)

    def loss(self, W):
        return np.sum(W)





