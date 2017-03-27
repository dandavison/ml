from collections import Counter
from random import shuffle

import numpy as np
import pandas as pd
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


def starmin(iterable, key=None):
    """
    Mitigate sadness about removal of tuple-unpacking in python3.
    """
    return min(iterable, key=(lambda item: key(*item)) if key else None)


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


def one_hot_encode_categorical_columns(df, max_n_categories):
    categorical_column_names = df.columns[df.dtypes == np.dtype('object')]
    for column_name in categorical_column_names:
        n_categories = len(set(df[column_name]))
        encoded = pd.get_dummies(df[column_name])
        del df[column_name]
        if n_categories > max_n_categories:
            print("%s: %d categories: dropping" % (column_name, n_categories))
            continue
        else:
            df = pd.concat([df, encoded], axis=1)
            print("%s: %d categories" % (column_name, n_categories))

    return df


def remove_rows_with_nulls_in_mostly_non_null_columns(df, max_nulls):
    column_null_counts = df.isnull().apply(sum, axis=0)
    columns_with_few_nulls = column_null_counts[column_null_counts <= max_nulls].index
    rows_with_nulls = (df.ix[:, columns_with_few_nulls].isnull()
                       .apply(sum, axis=1) > 0)
    return df.ix[~rows_with_nulls, :]


def get_X_y(df, y_column):
    if y_column in df:
        y = np.array(df[y_column])
        df = df.drop(y_column, axis=1)
        X = np.array(df)
        n, d = df.shape
        y = y.reshape((n, 1))
    else:
        X = np.array(df)
        y = None

    return X, y
