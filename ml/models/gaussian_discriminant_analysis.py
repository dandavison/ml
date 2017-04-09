from collections import Counter

import numpy as np
from numpy.linalg import det
from numpy.linalg import pinv as inv

from ml.models import Gaussian
from ml.models.base import Classifier
from ml.utils import split


__all__ = ['LinearDiscriminantAnalysis', 'QuadraticDiscriminantAnalysis']


class GaussianDiscriminantAnalysis(Classifier):
    """
    Abstract parent class for LDA and QDA.
    """
    method = None

    def fit(self, X, labels):
        """
        Estimate parameters of the model, and prior on classes.

        For LDA, parameters are within-class mean vectors mu, and a shared
        covariance matrix Sigma.

        For QDA, parameters are within-class mean vectors mu, and a
        covariance matrix Sigma for each class.
        """
        n, d = X.shape
        if self.method == 'LDA':
            self.Sigma = self._estimate_pooled_within_class_covariance(X, labels)  # noqa
            self.Sigma_inv = inv(self.Sigma)
        elif self.method == 'QDA':
            self.Sigma = [
                Gaussian().fit(X_c).Sigma
                for X_c in split(X, labels)
            ]
            self.Sigma_inv = list(map(inv, self.Sigma))
            self.Sigma_det = list(map(det, self.Sigma))
        else:
            raise ValueError("Invalid method: %s" % self.method)

        self.mus = np.array([
            Gaussian().fit(X_c).mu
            for X_c in split(X, labels)
        ])
        self.prior = {
            c: n_c / n
            for c, n_c in Counter(labels).items()
        }
        self.label_set = np.array(sorted(set(labels)))
        return self

    def predict(self, X):
        """
        Return predicted class labels for sample points (rows of X).
        """
        print("predict( %s )" % (X.shape,))
        discriminant_values = []
        for i, x in enumerate(X):
            if i and not i % 1000:
                print("%d" % i)
            row = [
                self.discriminant_function(x, c)
                for c in self.label_set
            ]
            discriminant_values.append(row)
        discriminant_values = np.array(discriminant_values)
        return self.label_set[discriminant_values.argmax(axis=1)]


class LinearDiscriminantAnalysis(GaussianDiscriminantAnalysis):
    method = 'LDA'

    def discriminant_function(self, x, c):
        """
        Compute discriminant function value for label c on sample point x.
        """
        c_index = list(self.label_set).index(c)
        mu = self.mus[c_index]
        prior = self.prior[c]
        Sigma_inv = self.Sigma_inv
        return (
            mu.T @ Sigma_inv @ x -
            mu.T @ Sigma_inv @ mu +
            np.log(prior)
        )

    def _estimate_pooled_within_class_covariance(self, X, labels):
        """
        Return pooled within-class covariance

        Return average over classes of (X_c^T X_c), where X_c is centered data
        for class C.
        """
        # Compute sum of within-class variances, each one multiplied by its
        # class sample size. Finally divide by total sample size.
        return sum(
            Gaussian().fit(X_c).Sigma * X_c.shape[0]
            for X_c in split(X, labels)
        ) / X.shape[0]


class QuadraticDiscriminantAnalysis(GaussianDiscriminantAnalysis):
    method = 'QDA'

    def discriminant_function(self, x, c):
        """
        Compute discriminant function value for label c on sample point x.
        """
        c_index = list(self.label_set).index(c)
        mu = self.mus[c_index]
        prior = self.prior[c]
        Sigma = self.Sigma[c_index]
        Sigma_det = self.Sigma_det[c_index]
        Sigma_inv = self.Sigma_inv[c_index]
        return (
            -1/2 * (x - mu).T @ Sigma_inv @ (x - mu) +
            -1/2 * np.log(Sigma_det) +
            np.log(prior)
        )
