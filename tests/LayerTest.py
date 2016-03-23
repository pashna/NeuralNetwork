import unittest
from Engine.Layer import Layer
import numpy as np

class LayerTest(unittest.TestCase):

    def test_random_init(self):
        layer = Layer(3, 5)


    def test_etalon_vector(self):
        layer = Layer(2, 4, label=[0,1,2,3])
        layer.propagate(np.asarray([3,1]))
        self.assertTrue( (np.asarray([0,0,1,0]) == layer.get_etalon_vector(2)).all())