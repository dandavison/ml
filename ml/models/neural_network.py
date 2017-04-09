import numpy as np

from ml.models.base import Classifier


__all__ = ['SingleLayerTanhLogisticNeuralNetwork']


class NeuralNetwork(Classifier):
    def fit(self, n_iter=10):
        for it in range(n_iter):
            self.forwards()
            self.backwards()
            for layer in self.layers:
                self.layer.W -= self.learning_rate * self.layer.gradient()

    def predict(X):
        Z = X
        for layer in self.layers:
            Z = layer.f(Z @ layer.W)
        return self.prediction_fn(Z)

    def prediction_fn(self, yhat):
        """
        Map values in output layer to classification predictions.
        """
        raise NotImplementedError


class Layer:
    """
    Each layer has two attributes:

    - W  a (j x k) weight matrix, where j is the number of units in the previous
         layer and k is the number of units in this layer.

    - f  activation function.

    The data values Z in the previous layer have dimension (n x j).
    The data values Z' in this layer have dimension (n x k).
    Z' is computed as

    Z' = f(ZW).
    """
    def __init__(self, activation_fn, weights_matrix):
        self.f = activation_fn
        self.W = weights_matrix


class LogisticRegressionNeuralNetwork(NeuralNetwork):
    """
    Logistic regression implemented as a neural network.
    """
    def __init__(self):
        self.n_hidden_layers = 0

    def prediction_fn(self, y_hat):
        return np.array(y_hat > 0.5, dtype=np.int)
