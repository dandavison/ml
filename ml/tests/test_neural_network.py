import random
from sys import stdout

import numpy as np
from numpy import array
from numpy import nan
from sklearn.neural_network.multilayer_perceptron import MLPClassifier

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

        self.assertEqualArrays(
            net.locals['grad__z_h__v_h'],
            array([0, 0]),
        )

        self.assertEqualArrays(
            net.locals['grad__L__v_h'],
            array([0, 0]),
        )

    def test_2(self):
        self._do_test_2(n_hidden_units=1)

    def _do_test_2(self, n_hidden_units):
        Xy = array([
            [-1,   -1, 1],
            [ 0,    0, 1],
            [ 1,    1, 1],
            [ 99,  99, 2],
            [100, 100, 2],
            [111, 111, 2],
        ])
        X = Xy[:, :-1]
        y = Xy[:, -1]

        learning_rate = 0.01
        n_iterations = 1000

        rng_seed = 0
        random.seed(rng_seed)
        np.random.seed(rng_seed)

        net = SingleLayerTanhLogisticNeuralNetwork(
            n_hidden_units=n_hidden_units,
            learning_rate=learning_rate,
            outfile=None,
            n_iterations=n_iterations,
        )
        net.fit(X, y)
        self.assertEqualArrays(
            np.argmax(net.predict(X), axis=1) + 1,
            y,
        )
        Yhat = net.predict(X)
        _, Y = net.prepare_data(X, y)
        self.assertEqual(
            net.loss(Yhat, Y),
            1.7538333305760909,
        )
        # ~ 0.5s

        if False:
            sk_net = SKLearnNeuralNet(
                learning_rate_init=learning_rate,
                max_iter=n_iterations,
            )
            sk_net.fit(X, y)
            print(sk_net.predict(X))


class SKLearnNeuralNet(MLPClassifier):

    def __init__(self, learning_rate_init, max_iter):
        super(SKLearnNeuralNet, self).__init__(
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            # Fixed
            hidden_layer_sizes=(2,),
            activation='tanh',
            solver='sgd',
            alpha=0,
            batch_size=1,
            verbose=False,
        )


if __name__ == '__main__':
    TestNeuralNetwork().test_2()
