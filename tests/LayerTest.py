import unittest
from Engine.Layer import Layer


class LayerTest(unittest.TestCase):

    def test_random_init(self):
        layer = Layer(3, 5)
        print layer._w