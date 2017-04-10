import sys
from collections import Counter
from functools import partial
from random import sample

import numpy as np
from numpy import inf
from numpy import tanh
from scipy.stats import describe

from ml.models.base import Classifier
from ml.utils import log
from ml.utils import logistic
from ml.utils import one_hot_encode_array


__all__ = ['SingleLayerTanhLogisticNeuralNetwork']

describe = partial(describe, axis=None)
log = partial(log, check=False)
logistic = partial(logistic, check=True)

EPSILON = sys.float_info.epsilon


class NeuralNetwork(Classifier):

    def predict(X):
        Z = X
        for layer in self.layers:
            Z = layer.f(Z @ layer.W)
        return self.prediction_fn(Z)

    def fit(self, X, Y, n_iter=10):
        for it in range(n_iter):
            self.forwards()
            self.backwards()
            for layer in self.layers:
                self.layer.W -= self.learning_rate * self.layer.gradient()

    def prediction_fn(self, yhat):
        """
        Map values in output layer to classification predictions.
        """
        raise NotImplementedError


class SingleLayerTanhLogisticNeuralNetwork(NeuralNetwork):
    """
    A classification neural net with one hidden layer.

    The hidden layer uses the tanh activation function.
    The output layer uses the logistic activation function.

    Model:

    The input data are X (n x d) and Y (n x K). We use stochastic gradient
    descent, i.e. compute and update gradients for a single input row at a
    time, so in backpropagation we work with x (d x 1) and y (K x 1).

    | Input                | x            | d x 1  |
    | First weight matrix  | V            | H x d  |
    | Hidden layer         | Z = tanh(Vx) | H x 1  |
    | Second weight matrix | W            | K x H  |
    | Output               | yhat         | K x 1  |
    | Loss                 | L            | scalar |

    The loss function is the cross-entropy
    sum_k { y_k log(yhat_k) + (1 - y_k) log(1 - yhat_k) }
    """

    def __init__(self, n_hidden_units):
        self.H = n_hidden_units
        self.K = None  # Determined empirically as distinct training labels
        self.V = None
        self.W = None

    def predict(self, X):
        Z = tanh(self.V @ X.T)
        Yhat = logistic(self.W @ Z).T
        return Yhat

    def fit(self, X, y,
            learning_rate=1e-3,
            n_iterations=None,
            stop_factor=None,
            stop_window_size=100,
            outfile=None):
        """
        \grad_{W_k} L = \partiald{L}{\yhat_k} \grad_{W_k} \yhat_k
            \partiald{L}{\yhat_k} = \frac{y_k - \yhat_k}{\yhat_k (1 - \yhat_k)}
            \grad_{W_k} \yhat_k = z \yhat_k (1 - \yhat_k)

        \grad_z L = \sum_k \partiald{L}{\yhat_k} \grad_z \yhat_k
            \grad_z \yhat_k = W_k \yhat_k (1 - \yhat_k)

        \grad_{V_h} L = \partiald{L}{z_h} \grad_{V_h} z_h
            \grad_{V_h} z_h = x(1 - z_h^2)
        """
        assert stop_factor or n_iterations

        X, Y = self.prepare_data(X, y)
        H = self.H
        K = self.K

        n, d = X.shape

        V = self.V = np.random.uniform(-1, 1, size=H * d).reshape((H, d))
        W = self.W = np.random.uniform(-1, 1, size=K * H).reshape((K, H))

        Yhat = self.predict(X)

        L = self.loss(Yhat, Y)
        print('Loss: %.2f' % L)

        delta_L_window = np.zeros(stop_window_size)
        it = 0
        while True:

            if it % 1000 == 0:
                sys.stderr.write('%d\n' % it)

            i, = sample(range(n), 1)

            x = X[i, :]
            z = tanh(V @ x)
            yhat = Yhat[i, :]
            y = Y[i, :]
            L_i_before = self.loss(yhat, y)

            grad__L__yhat = (y - yhat) / np.clip((yhat * (1 - yhat)), EPSILON, inf)

            # Update W
            grad__L__z = np.zeros_like(z)
            for k in range(K):
                grad__yhat_k__W_k = z * yhat[k] * (1 - yhat[k])
                grad__yhat_k__z = W[k, :] * yhat[k] * (1 - yhat[k])
                grad__L__z += grad__L__yhat[k] * grad__yhat_k__z
                W[k, :] -= learning_rate * grad__L__yhat[k] * grad__yhat_k__W_k

            # Update V
            for h in range(H):
                grad__z_h__v_h = x * (1 - z[h] ** 2)
                grad__L__v_h = grad__L__z[h] * grad__z_h__v_h
                V[h, :] -= learning_rate * grad__L__v_h

            z = tanh(V @ x)
            yhat = logistic(W @ z)

            assert np.isfinite(yhat).all()

            Yhat[i, :] = yhat
            L_i_after = self.loss(yhat, y)
            assert np.isfinite(L_i_after)
            delta_L = L_i_after - L_i_before
            if not delta_L < 1e-3:
                sys.stderr.write("Δ L = %.2f\n" % delta_L)
            if outfile:
                outfile.write('%.2f\n' % delta_L)
                outfile.flush()
            delta_L_window[it % stop_window_size] = delta_L
            L += delta_L

            it += 1
            if n_iterations and it == n_iterations:
                break
            elif abs(delta_L_window[:it].mean()) < stop_factor:
                break


    def loss(self, Yhat, Y):
        log_Yhat = log(Yhat)
        log_Yhat_inv = log(1 - Yhat)

        log_Yhat[~np.isfinite(log_Yhat)] = log(EPSILON)
        log_Yhat_inv[~np.isfinite(log_Yhat_inv)] = log(EPSILON)

        return (Y * log_Yhat + (1 - Y) * log_Yhat_inv).sum()

    def prepare_data(self, X, y):
        n, d = X.shape
        nY, = y.shape
        assert nY == n
        self.K = len(set(y))
        # Demand that labels are integers 1...max(y)
        if not np.issubdtype(y.dtype, np.int):
            y_int = np.floor(y).astype(np.int)
            assert (y_int == y).all()
            y = y_int
        assert set(y) == set(np.arange(self.K) + 1)
        Y = one_hot_encode_array(y)
        return X, Y


class Layer:
    """
    Each layer has two attributes:

    - W a (j x k) weight matrix, where j is the number of units in the previous
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
