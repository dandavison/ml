import numpy as np
from numpy import array
from numpy import nan

from ml.models import SingleLayerTanhLogisticNeuralNetwork
from ml.tests import TestCase


class TestNeuralNetwork(TestCase):

    def test_1(self):
        """
        One 1-d sample point. One hidden layer.
        """
        X = array([[0]])
        y = array([1])
        net = SingleLayerTanhLogisticNeuralNetwork(n_hidden_units=1)
        net.V = array([[0.0, 0.0],
                       [nan, nan]])  # last row should be ignored
        net.W = array([[0.0, 0.0]])
        Yhat = net.predict(X)

        self.assertEqualArrays(
            Yhat,
            array([[0.5]]))

        net.fit(X, y, n_iterations=1)

        self.assertEqualArrays(
            net.locals['grad__yhat_k__W_k'],
            array([0, 0.25])
        )

        self.assertEqualArrays(
            net.locals['grad__L__yhat'],
            array([2]),
        )

        self.assertEqualArrays(
            net.locals['grad__yhat_k__z'],
            array([0, 0]),
        )

        self.assertEqualArrays(
            net.locals['grad__L__z'],
            array([0, 0]),
        )


if __name__ == '__main__':
    TestNeuralNetwork().test_1()
    print('OK')
