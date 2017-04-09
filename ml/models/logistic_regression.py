import sys

import numpy as np

from ml.models.base import Classifier


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


def logistic(z):
    p = 1 / (1 + np.exp(-z))
    assert np.isfinite(p).all()
    assert (0 < p).all() and (p < 1).all()
    return p


def log(x):
    log_x = np.log(x)
    assert np.isfinite(log_x).all()
    return log_x
