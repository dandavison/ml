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

    def predict(x):
        return self.tree.predict(x)


class DecisionTreeData(np.ndarray):

    @property
    def X(self):
        return self[:,:-1]

    @property
    def y(self):
        return self[:,-1]


class Node:

    def __init__(self, left=None, right=None, feature=None, decision_boundary=None, label=None):  # noqa
        self.left = None
        self.right = None
        self.feature = feature
        self.decision_boundary = None
        self.label = label

    @property
    def is_leaf(self):
        return self.label is not None

    def predict(self, x):
        if self.is_leaf:
            return self.label
        elif x[self.feature] <= self.decision_boundary:
            return self.left.predict(x)
        else:
            return self.right.predict(x)


def grow_tree(data):
    labels = set(y)
    if len(labels) == 1:
        [c] = labels
        return Node(label=c)
    else:
        left, right, feature, decision_boundary = choose_partition(data)
        return Node(grow_tree(left),
                    grow_tree(right),
                    feature,
                    decision_boundary)


def choose_partition(data):
    n, d = data.X.shape
    partition, j, decision_boundary, score = max(
        ((partition, j, decision_boundary, score)
         for j in range(d)
         for (partition, decision_boundary, score) in get_feature_partitions(data[:, [j, -1]])),
        key=lambda partition, j, decision_boundary, score: score)
    left, right = data[~partition, :], data[partition, :]
    return left, right, j, decision_boundary


def get_feature_partitions(data):
    """
    Generator yielding (partition, score) tuples.

    `data` is an n x 2 array containing a single feature column, and the labels.

    Sort the feature values. Walk left to right through the sorted feature
    values. At each transition, count the distribution of labels to the left
    and right. Return the partition induced by that cut point and the weighted
    average of the entropies of the left and right label distributions.

    >>> from numpy import log2
    >>> approximately_equal = lambda x, y: abs(x - y) < 1e-15
    >>> data = np.array([[1.2, 0.7, 0.7, 2.2, 5.1, 6.2], [0, 0, 0, 1, 0, 1]]).T
    >>> partitions = get_feature_partitions(data)
    >>> partition, score, decision_boundary = next(partitions)
    >>> decision_boundary == 0.7
    True
    >>> list(partition)  # Note that the first row comes after the second and third when sorted by feature value  # noqa
    [True, False, False, True, True, True]
    >>> # |         | left | right |
    >>> # |---------+------+-------|
    >>> # | counts  |  2,0 | 2, 2  |
    >>> # | weight  |  2/6 | 4/6   |
    >>> # | entropy |    0 | 1     |
    >>> score == 2/3
    True
    >>> partition, score, decision_boundary = next(partitions)
    >>> decision_boundary == 1.2
    True
    >>> list(partition)
    [False, False, False, True, True, True]
    >>> # |         | left | right                              |
    >>> # |---------+------+------------------------------------|
    >>> # | counts  | 3,0  | 1, 2                               |
    >>> # | weight  | 3/6  | 3/6                                |
    >>> # | entropy | 0    | -(1/3)*log2(1/3) - (2/3)*log2(2/3) |
    >>> approximately_equal(score, -(1/2)*( (1/3)*log2(1/3) + (2/3)*log2(2/3)))
    True
    """
    n, d = data.shape
    assert n > 1 and d == 2
    counts = Counter(data[:, 1])
    sort_permutation = data[:, 0].argsort()
    data = data[sort_permutation, :]

    n, _ = data.shape

    left_counts = Counter()

    decision_boundary, _ = data[0]
    for i, (x, y) in enumerate(data):
        if x > decision_boundary:
            partition = np.zeros(n, dtype=np.bool)
            partition[sort_permutation[i:]] = True
            score = weighted_average_entropy(left_counts, counts - left_counts)
            yield partition, score, decision_boundary
            decision_boundary = x
        left_counts[y] += 1


def weighted_average_entropy(*counters):
    size = lambda counter: sum(counter.values())
    n_total = sum(size(c) for c in counters)
    return sum(
        (size(c) * entropy(c.values()))
        for c in counters
    ) / n_total
