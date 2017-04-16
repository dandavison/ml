from __future__ import print_function
import sys
import re
from collections import Counter
from functools import partial
from random import shuffle
from sys import stderr

import numpy as np
from numpy import diag
from numpy import inf
from numpy import tanh
from scipy.stats import describe

from ml.models.base import Classifier
from ml.utils import cyclic
from ml.utils import log
from ml.utils import logistic
from ml.utils import memoized
from ml.utils import nans_like
from ml.utils import one_hot_encode_array
from ml.utils import random_normal

from clint.textui.colored import red, blue, green
red = partial(red, bold=True)
blue = partial(blue, bold=True)
green = partial(green, bold=True)
COLOURS = cyclic([green, blue, red])
DEBUG = False
QUIET = True
USE_NUMERICAL_DERIVATIVES = False


@memoized
def get_colour(key):
    return next(COLOURS)


__all__ = ['SingleLayerTanhLogisticNeuralNetwork']

describe = partial(describe, axis=None)
log = partial(log, check=False)
logistic = partial(logistic, check=True)

EPSILON = sys.float_info.epsilon
EPSILON_FINITE_DIFFERENCE = 1e-6


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
    -sum_k { y_k log(yhat_k) + (1 - y_k) log(1 - yhat_k) }
    """

    def __init__(self,
                 n_hidden_units,
                 learning_rate,
                 n_iterations=None,
                 stop_factor=None,
                 stop_window_size=None,
                 outfile=None):

        if stop_window_size:
            assert DEBUG

        self.H = n_hidden_units
        self.K = None  # Determined empirically as distinct training labels
        self.V = None
        self.W = None
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.stop_factor = stop_factor
        self.stop_window_size = stop_window_size
        self.outfile = outfile

    def predict(self, X):
        X = self.prepare_data(X)
        Z, Yhat = self.forward(X, self.V, self.W)
        return Yhat

    def forward(self, X, V, W):
        Z = tanh(V @ X.T)
        Z[-1, :] = 1  # The last row of V is unused; z[-1] must always be 1, just as x[-1].
        Yhat = logistic(self.W @ Z).T
        return Z, Yhat

    def fit(self, X, y):
        """
        \grad_{W_k} L = \partiald{L}{\yhat_k} \grad_{W_k} \yhat_k
            \partiald{L}{\yhat_k} = \frac{y_k - \yhat_k}{\yhat_k (1 - \yhat_k)}
            \grad_{W_k} \yhat_k = z \yhat_k (1 - \yhat_k)

        \grad_z L = \sum_k \partiald{L}{\yhat_k} \grad_z \yhat_k
            \grad_z \yhat_k = W_k \yhat_k (1 - \yhat_k)

        \grad_{V_h} L = \partiald{L}{z_h} \grad_{V_h} z_h
            \grad_{V_h} z_h = x(1 - z_h^2)
        """
        assert self.stop_factor or self.n_iterations is not None

        X, Y = self.prepare_data(X, y)
        H = self.H
        K = self.K

        n, d = X.shape
        # X has extra offset dimension containing all 1s
        # The hidden layer z also has a unit whose value is always 1
        d -= 1

        if self.V is None:
            self.V = random_normal(0, 0.1, (H + 1, d + 1))
        if self.W is None:
            self.W = random_normal(0, 0.1, (K, H + 1))
        V, W = self.V, self.W

        Yhat = self.predict(X[:, :-1])
        # Allocate
        grad__L__z = np.zeros((H,))
        sample_indices = list(range(n))
        shuffle(sample_indices)

        delta_L_window = np.zeros(self.stop_window_size)
        it = -1
        while True:
            it += 1

            if it >= self.n_iterations:
                break
            if it % 10000 == 0:
                print('%6d/%-6d %.3f' % (it, self.n_iterations, self.loss(Yhat, Y)))

            i = sample_indices[it % n]

            x = X[[i], :]
            y = Y[i, :]
            z, yhat = self.forward(x, V, W)
            z = z.ravel()

            yhat = Yhat[i, :]

            # Update W
            # grad__L__yhat = (yhat - y) / np.clip(yhat * (1 - yhat), EPSILON, inf)
            # grad__L__z[:] = 0.0
            # for k in range(K):
            #     grad__yhat_k__W_k = z * yhat[k] * (1 - yhat[k])
            #     # Last element corresponds to constant offset 1 appended to z
            #     # vector; it does not change / has no derivative.
            #     grad__yhat_k__z = W[k, :-1] * yhat[k] * (1 - yhat[k])
            #     grad__L__z += grad__L__yhat[k] * grad__yhat_k__z
            #     W[k, :] -= self.learning_rate * grad__L__yhat[k] * grad__yhat_k__W_k
            grad__L__z = (W.T * (yhat - y)).sum(axis=1)
            zz = z.reshape((1, H + 1)).repeat(K, 0)
            grad__L__W = diag(yhat - y) @ zz
            W -= self.learning_rate * grad__L__W

            # Update V
            # for h in range(H):
            #     grad__z_h__V_h = x * (1 - z[h] ** 2)
            #     grad__L__V_h = grad__L__z[h] * grad__z_h__V_h
            #     V[h, :] -= self.learning_rate * grad__L__V_h
            xx = x.reshape((1, d + 1)).repeat(H + 1, 0)
            grad__L__V = diag((1 - z ** 2) * grad__L__z) @ xx
            V -= self.learning_rate * grad__L__V

            z, yhat = self.forward(x, V, W)

            Yhat[i, :] = yhat

        return self

    def estimate_grad__z_h__V_h(self, h, x, V, grad):
        eps = EPSILON_FINITE_DIFFERENCE
        def d(j):
            eps_vec = np.zeros_like(V)
            eps_vec[h, j] = eps
            z_plus = tanh((V + eps_vec) @ x)
            z_minus = tanh((V - eps_vec) @ x)
            z_plus[-1] = 1
            z_minus[-1] = 1
            return (z_plus[h] - z_minus[h]) / (2 * eps)
        return self._do_finite_difference_estimate(
            d,
            V[h, :],
            'grad__z[%d]__V[%d,:]' % (h, h),
            grad,
        )

    def estimate_grad__yhat_k__z(self, k, z, W, y, grad):
        eps = EPSILON_FINITE_DIFFERENCE
        def d(h):
            eps_vec = np.zeros_like(z)
            eps_vec[h] = eps
            yhat_plus = logistic(W @ (z + eps_vec))
            yhat_minus = logistic(W @ (z - eps_vec))
            return (yhat_plus[k] - yhat_minus[k]) / (2 * eps)
        return self._do_finite_difference_estimate(
            d,
            z,
            'grad__yhat[%d]__z' % k,
            grad,
        )

    def estimate_grad__yhat_k__W_k(self, k, z, W, y, grad):
        eps = EPSILON_FINITE_DIFFERENCE
        def d(h):
            eps_vec = np.zeros_like(W)
            eps_vec[k, h] = eps
            yhat_plus = logistic((W + eps_vec) @ z)
            yhat_minus = logistic((W - eps_vec) @ z)
            return (yhat_plus[k] - yhat_minus[k]) / (2 * eps)
        return self._do_finite_difference_estimate(
            d,
            W[k, :],
            'grad__yhat[%d]__W[%d,:]' % (k, k),
            grad,
        )

    def estimate_grad__L__yhat(self, yhat, y, grad):
        eps = EPSILON_FINITE_DIFFERENCE
        def d(k):
            eps_vec = np.zeros_like(yhat)
            eps_vec[k] = eps
            L_plus = self.loss(yhat + eps_vec, y)
            L_minus = self.loss(yhat - eps_vec, y)
            return (L_plus - L_minus) / (2 * eps)
        return self._do_finite_difference_estimate(
            d,
            yhat,
            'grad__L__yhat',
            grad,
        )

    def estimate_grad__L__z(self, z, W, y, grad):
        eps = EPSILON_FINITE_DIFFERENCE
        def d(h):
            eps_vec = np.zeros_like(z)
            eps_vec[h] = eps
            yhat_plus = logistic(W @ (z + eps_vec))
            yhat_minus = logistic(W @ (z - eps_vec))
            L_plus = self.loss(yhat_plus, y)
            L_minus = self.loss(yhat_minus, y)
            return (L_plus - L_minus) / (2 * eps)
        return self._do_finite_difference_estimate(
            d,
            z,
            'grad__L__z',
            grad,
        )

    def estimate_grad__L__V_h(self, h, x, V, W, y, grad):
        eps = EPSILON_FINITE_DIFFERENCE
        def d(j):
            eps_vec = np.zeros_like(V)
            eps_vec[h, j] = eps
            z_plus = tanh((V + eps_vec) @ x)
            z_minus = tanh((V - eps_vec) @ x)
            z_plus[-1] = 1
            z_minus[-1] = 1
            yhat_plus = logistic(W @ z_plus)
            yhat_minus = logistic(W @ z_minus)
            L_plus = self.loss(yhat_plus, y)
            L_minus = self.loss(yhat_minus, y)
            return (L_plus - L_minus) / (2 * eps)
        return self._do_finite_difference_estimate(
            d,
            V[h, :],
            'grad__L__V_h',
            grad,
        )

    @staticmethod
    def _do_finite_difference_estimate(d, wrt, label, grad):
        grad__n = np.array(list(map(d, range(len(wrt)))))
        if DEBUG:
            col = get_colour(re.subn(r'\d+', '%d', label))
            print(col('%s = %s' % (label, grad__n)))
            print(col(', '.join('%.9f' % g for g in describe(grad__n - grad).minmax)))
        return grad__n

    def compute_loss(self, X, Y):
        Yhat = self.predict(X)
        return self.loss(Yhat, Y)

    def loss(self, Yhat, Y):
        log_Yhat = log(Yhat)
        log_Yhat_inv = log(1 - Yhat)

        log_Yhat[Y == 0] = 0
        log_Yhat_inv[Y == 1] = 0
        if not (np.isfinite(log_Yhat).all() and
                np.isfinite(log_Yhat_inv).all()):
            stderr.write('parameters incompatible with data '
                         '(log() arising in loss calculation).\n')
            stderr.flush()

        log_Yhat[~np.isfinite(log_Yhat)] = log(EPSILON)
        log_Yhat_inv[~np.isfinite(log_Yhat_inv)] = log(EPSILON)

        return -(Y * log_Yhat + (1 - Y) * log_Yhat_inv).sum()

    def prepare_data(self, X, y=None):
        n, d = X.shape
        X = np.hstack([X, np.ones((n, 1))])
        if y is None:
            return X

        nY, = y.shape
        assert nY == n
        K = len(set(y))
        # Demand that labels are integers 1...max(y)
        if not np.issubdtype(y.dtype, np.int):
            y_int = np.floor(y).astype(np.int)
            assert (y_int == y).all()
            y = y_int

        assert set(y) == set(np.arange(K) + 1), \
            'Some labels are not represented in training data'

        self.K = K
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
