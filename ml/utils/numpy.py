from random import shuffle

import numpy as np
from numpy import log2


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


def logistic(z, check=True):
    p = 1 / (1 + np.exp(-z))
    if check:
        assert np.isfinite(p).all()
        assert (0 <= p).all() and (p <= 1).all()
    return p


def log(x, check=True):
    log_x = np.log(x)
    if check:
        assert np.isfinite(log_x).all()
    return log_x


def multiply_with_zeros_and_nonfinite_values(a, b):
    prod = a * b
    prod[a == 0] = 0
    prod[b == 0] = 0

    if not np.isfinite(prod).all():
        import ipdb ; ipdb.set_trace()

    return prod


def stop_if_nonfinite(x):
    if np.isfinite(x).all():
        return x
    import sys
    import ipdb
    sys.stderr.write('Non-finite values found\n')
    ipdb.set_trace()


def random_uniform(low, high, shape):
    return (np.random.uniform(low, high, size=np.prod(shape))
            .reshape(shape))


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


def inverse_permutation(x):
    """
    Return the indices that arrange x in sorted order.

    Same as np.argsort(x), but O(n) instead of O(n log(n)).

    http://arogozhnikov.github.io/2015/09/29/NumpyTipsAndTricks1.html
    """
    px = np.empty(len(x), dtype=np.int)
    px[x] = range(len(x))
    return px


def entropy(counts):
    """
    Entropy of probability distribution estimated from counts.

    -\sum p_i log p_i = -\sum (n_i/N) log (n_i/N)
                      = log N - (1/N) \sum n_i log n_i

    >>> entropy([]) == entropy([1]) == 0
    True
    >>> entropy([1, 1]) == entropy([2, 2]) == 1
    True
    >>> from math import log2
    >>> entropy([1, 1, 1]) == entropy([2, 2, 2]) == log2(3)
    True
    """
    n_total = sum(counts)
    if False:
        pp = [n / n_total for n in counts]
        return -sum(p * log2(p) for p in pp)
    else:
        if n_total == 0:
            return 0.0
        else:
            return log2(n_total) - sum(n * log2(n) for n in counts) / n_total


def one_hot_encode_array(x):
    # 1D vector of positive integers
    assert (len(x.shape) == 1 and
            (x > 0).all() and
            np.issubdtype(x.dtype, np.int))
    encoded = np.zeros((x.size, x.max()), dtype=np.int)
    encoded[np.arange(x.size), x - 1] = 1
    return encoded
