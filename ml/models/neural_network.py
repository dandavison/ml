import numpy as np


class NeuralNetwork(Classifier):
    def fit(self, n_iter=10):
        for it in range(n_iter):
            self.params -= self.learning_rate * self.gradient()

    def predict(X):
        yhat = X
        for l in self.layers:
            yhat = l.f(yhat @ l.W)
        return self.prediction_fn(yhat)

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

    The data values \hat{Y} in the previous layer have dimension (n x j).
    The data values \hat{Y'} in this layer have dimension (n x k).
    The data values \hat{Y'} in this layer are computed as

    \hat{Y'} = f(\hat{Y} W).

    The data values are named \hat{Y} because they are a matrix at some point in
    the transformation of the input data X to the predictions \hat{Y}.
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
