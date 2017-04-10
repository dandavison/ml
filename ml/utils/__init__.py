from ml.utils.numpy import *


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
