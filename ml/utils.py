from random import shuffle

import numpy as np


def contrast_normalize(X):
    """
    Return input data normalized by dividing by the l2-norm of each row.
    """
    n, d = X.shape
    l2_norm = np.sqrt((X ** 2).sum(1)).reshape((n, 1))
    return X / l2_norm


def normalize_features_to_unit_length(X):
    # Divide each column by its l2-norm
    return X @ np.diag(1. / np.sqrt((X ** 2).sum(axis=0)))


def normalize_features_to_unit_variance(X):
    # Divide each column by its standard deviation

    # This should be the same as normalizing to unit length if the data is
    # centered, so maybe
    # return mean(X) + normalize_features_to_unit_length(center(X))
    return X @ np.diag(1 / np.sqrt(var(X)))


def center(X):
    """
    Transform X so that all columns have mean zero.
    """
    return X - X.mean(axis=0)


def cov(X):
    """
    Covariance matrix of columns of X.

    jk-th entry of result is

    \frac{1}{n} \sum_i (X_{ij} - \mu_j)(X_{ik} - \mu_k)
    """
    # 3 equivalent implementations:
    if False:
        return np.cov(X, rowvar=False, bias=True)
    elif False:
        n, d = X.shape
        mu = X.mean(axis=0).reshape((d, 1))
        return (1/n) * X.T @ X - mu @ mu.T
    else:
        n, d = X.shape
        X = center(X)
        return (1/n) * X.T @ X


def var(X):
    """
    Array of variances of columns of X.
    """
    if False:
        return np.var(X, axis=0)
    else:
        return np.diag(cov(X))


def l2_norm(x):
    return np.sqrt((x ** 2).sum())


def split(X, labels):
    """
    A generator yielding subsets of data defined by the labels.
    """
    for label in sorted(set(labels)):
        yield X[labels == label, :]


def mean(iterable):
    n = 0
    total = 0
    for el in iterable:
        total += el
        n += 1
    return total / n


def random_partition(X, y, n):
    """
    Split into n rows and remaining rows, after shuffling.
    """
    n_total, d = X.shape
    row_indices = list(range(n_total))
    shuffle(row_indices)
    return (
        X[row_indices[:n]],
        y[row_indices[:n]],
        X[row_indices[n:]],
        y[row_indices[n:]],
    )
