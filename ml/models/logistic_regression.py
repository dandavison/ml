import sys
from functools import partial

import numpy as np

from ml.models.base import Classifier
from ml.utils import log
from ml.utils import logistic


log = partial(log, check=True)
logistic = partial(logistic, check=True)


__all__ = ['LogisticRegressionWithL2Regularization']


class LogisticRegressionWithL2Regularization(Classifier):
    """
    Cost function is J(w) = -l(w) + \lambda |w|^2

    l(w) = loglikelihood(w)
    """

    def __init__(self, _lambda, epsilon):
        """
        epsilon -- learning rate
        """
        self.epsilon = epsilon
        self._lambda = _lambda
        self.w = None

    def fit(self, X, y, n_iter=100, verbose=False):
        """
        Gradient descent
        """
        n, d = X.shape
        w = np.zeros((d, 1))  # start with all weights zero and zero offset

        for it in range(n_iter):
            if verbose:
                print(', '.join('% 11.2f' % _w for _w in w))
                print(self.loglike(w, X, y))
            w = w - self.epsilon * self.gradient(X, y, w)

        self.w = w
        return self

    def predict(self, X):
        prob = logistic(X @ self.w)
        return np.array(prob > 0.5, dtype=np.int)

    def loglike(self, w, X, y):
        Xw = X @ w
        return (
            log(     logistic(Xw)  **      y) +
            log((1 - logistic(Xw)) ** (1 - y))
        ).sum()

    def cost(self, w, X, y):
        return self._lambda * (w ** 2).sum() - self.loglike(w, X, y)

    def gradient(self, X, y, w):
        return 2 * self._lambda * w - X.T @ (y - logistic(X @ w))


def logistic_regression_newton_update(w, X, y, _lambda):
    s = logistic(X @ w)
    gradient = 2 * _lambda * w - X.T @ (y - s)
    B = np.diag((s * (1 - s) + 2 * _lambda).ravel())
    hessian = X.T @ B @ X
    return w - np.inv(hessian) @ gradient
