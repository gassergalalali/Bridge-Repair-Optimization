"""
This is a unittest for the model
"""

import unittest

from .Model import Model


class TestStringMethods(unittest.TestCase):
    def test_model(self):
        """
        Just test the model is working
        """
        ga_max_epochs = 5
        ga_population_size = 15
        model = Model(ga_max_epochs, ga_population_size)
        model.ga_run()


if __name__ == '__main__':
    unittest.main()
