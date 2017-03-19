from collections import Counter

import numpy as np
from numpy import log2

from ml.utils import entropy
from ml.utils import inverse_permutation
from ml.utils import mean


class DecisionTree:

    def __init__(self):
        self.tree = None

    def fit(X, y):
        data = np.hstack([X, y]).view(DecisionTreeData)
        self.tree = grow_tree(data)


class DecisionTreeData(np.ndarray):

    @property
    def X(self):
        return self[:,:-1]

    @property
    def y(self):
        return self[:,-1]


class Node:

    def __init__(self, left=None, right=None, feature=None, label=None):
        self.left = None
        self.right = None
        self.feature = feature
        self.label = label


def grow_tree(data):
    labels = set(y)
    if len(labels) == 1:
        [c] = labels
        return Node(label=c)
    else:
        left, right, feature = choose_partition(data)
        return Node(grow_tree(left),
                    grow_tree(right),
                    feature)


def choose_partition(data):
    n, d = data.X.shape
    partition, j, score = max(
        ((partition, j, score)
         for j in range(d)
         for (partition, score) in get_feature_partitions(data[:, [j, -1]])),
        key=lambda partition, j, score: score)
    left, right = data[~partition, :], data[partition, :]
    return left, right, j


def get_feature_partitions(data):
    """
    Generator yielding (partition, score) tuples.

    `data` is an n x 2 array containing a single feature column, and the labels.

    Sort the feature values. Walk left to right through the sorted feature
    values. At each transition, count the distribution of labels to the left
    and right. Return the partition induced by that cut point and the weighted
    average of the entropies of the left and right label distributions.

    >>> from numpy import log2
    >>> data = np.array([[1.2, 0.7, 2.2, 5.1, 6.2], [0, 0, 1, 0, 1]]).T
    >>> partitions = get_feature_partitions(data)
    >>> partition, score = next(partitions)
    >>> list(partition)  # Note that the first two rows are switched when sorted by feature value  # noqa
    [True, False, True, True, True]
    >>> # |         | left | right |
    >>> # |---------+------+-------|
    >>> # | counts  |  1,0 | 2, 2  |
    >>> # | weight  |  1/5 | 4/5   |
    >>> # | entropy |    0 | 1     |
    >>> score == 0.8
    True
    >>> partition, score = next(partitions)
    >>> list(partition)
    [False, False, True, True, True]
    >>> # |         | left | right                              |
    >>> # |---------+------+------------------------------------|
    >>> # | counts  | 2,0  | 1, 2                               |
    >>> # | weight  | 2/5  | 3/5                                |
    >>> # | entropy | 0    | -(1/3)*log2(1/3) - (2/3)*log2(2/3) |
    >>> score == -(3/5)*( (1/3)*log2(1/3) + (2/3)*log2(2/3) )
    True
    """
    n, d = data.shape
    assert n > 1 and d == 2
    counts = Counter(data[:, 1])
    sort_permutation = data[:, 0].argsort()
    data = data[sort_permutation, :]
    inv_sort_permutation = inverse_permutation(sort_permutation)

    n, _ = data.shape

    left_counts = Counter()

    curr, _ = data[0]
    for i, (x, y) in enumerate(data):
        if x > curr:
            partition = np.zeros(n, dtype=np.bool)
            partition[inv_sort_permutation[i:]] = True
            score = weighted_average_entropy(left_counts, counts - left_counts)
            yield partition, score
        left_counts[y] += 1


def weighted_average_entropy(*counters):
    size = lambda counter: sum(counter.values())
    n_total = sum(size(c) for c in counters)
    return sum(
        (size(c) * entropy(c.values()))
        for c in counters
    ) / n_total
