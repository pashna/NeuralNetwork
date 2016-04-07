import numpy as np

class L2():

    def derivative_matrix(self, W):
        return W

    def loss(self, W):
        return np.sum(np.square(W))