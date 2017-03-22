from collections import Counter
from functools import partial

import numpy as np
import pygraphviz as pgv
from numpy import log2

from ml.models import Classifier
from ml.utils import entropy
from ml.utils import inverse_permutation
from ml.utils import mean
from ml.utils import starmin


VERBOSE = False

class DecisionTree(Classifier):

    def __init__(self, feature_names=None, max_depth=np.inf):
        self.tree = None
        self.feature_names = feature_names
        self.max_depth = max_depth

    def fit(self, X, y):
        data = np.hstack([X, y]).view(DecisionTreeData)
        if VERBOSE:
            print("Fitting decision tree: %d observations x %d features" % data.shape)
        node_factory = partial(Node, feature_names=self.feature_names)
        self.tree = grow_tree(data, node_factory, max_depth=self.max_depth)

    def predict(self, X):
        return np.array([self.tree.predict(x) for x in X])

    def describe(self):
        self.tree.describe()

    def write_png(self, filename):
        graph = pgv.AGraph(directed=True)
        self.tree.add_to_graph(graph)
        graph.layout('dot')
        graph.draw(filename)


class DecisionTreeData(np.ndarray):

    @property
    def X(self):
        return self[:,:-1]

    @property
    def y(self):
        return self[:,-1]


class Node:

    def __init__(self,
                 left=None,
                 right=None,
                 feature=None,
                 decision_boundary=None,
                 label=None,
                 counts=False,
                 feature_names=None):

        self.left = left
        self.right = right
        self.feature = feature
        self.decision_boundary = decision_boundary
        self.label = label
        self.counts = counts
        self.feature_names = feature_names

    @property
    def is_leaf(self):
        return self.label is not None

    @property
    def feature_name(self):
        if self.feature is None:
            return None
        elif self.feature_names is not None:
            return self.feature_names[self.feature]
        else:
            return 'feature %d' % self.feature

    def predict(self, x):
        if self.is_leaf:
            return self.label
        elif x[self.feature] <= self.decision_boundary:
            return self.left.predict(x)
        else:
            return self.right.predict(x)

    def __repr__(self):
        context = dict(vars(self),
                       self_id=id(self),
                       left_id=id(self.left),
                       right_id=id(self.right),
                       feature_name=self.feature_name)
        if self.is_leaf:
            return 'Node {self_id}: class {label}'.format(**context)
        else:
            return (
                'Node {self_id}: if {feature_name} <= {decision_boundary}, '
                'go to node {left_id}, else {right_id}.'.format(**context))

    def describe(self):
        print(self)
        if not self.is_leaf:
            self.left.describe()
            self.right.describe()

    def add_to_graph(self, graph):
        """
        Add tree rooted at this node to pygraphviz graph.
        """
        graph.add_node(self)
        pgv_node = graph.get_node(self)
        if self.is_leaf:
            label = self.label
        else:
            label = '{feature_name} <= {decision_boundary}?'.format(
                feature_name=self.feature_name,
                **vars(self))
        pgv_node.attr.update(label=label)
        if self.left:
            self.left.add_to_graph(graph)
            graph.add_edge(self, self.left, label='No')
        if self.right:
            self.right.add_to_graph(graph)
            graph.add_edge(self, self.right, label='Yes')


def grow_tree(data, node_factory, depth=0, max_depth=float('inf')):
    label_counts = Counter(data.y)
    labels = label_counts.keys()
    predict = lambda: label_counts.most_common()[0][0]
    if len(labels) == 1 or depth > max_depth:
        node = node_factory(label=predict(), counts=label_counts)
    else:
        partition = choose_partition(data)
        left, right = data[partition.partition, :], data[~partition.partition, :]
        n_right, d = right.shape
        if n_right == 0:
            # All features are constant on this subset of sample points
            node = node_factory(label=predict(), counts=label_counts)
        else:
            node = node_factory(grow_tree(left, node_factory, depth + 1, max_depth=max_depth),
                                grow_tree(right, node_factory, depth + 1, max_depth=max_depth),
                                partition.feature,
                                partition.decision_boundary)
    if VERBOSE:
        print(node, flush=True)

    return node



class Partition:
    def __init__(self, decision_boundary, partition, score, feature=None):
        self.decision_boundary = decision_boundary
        self.partition = partition
        self.score = score
        self.feature = feature


def choose_partition(data):
    n, d = data.X.shape
    j, partition = starmin(
        ((j, partition)
         for j in range(d)
         for partition in get_feature_partitions(data[:, [j, -1]])),
        key=lambda j, partition: partition.score)
    partition.feature = j
    return partition


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
    >>> data = data.view(DecisionTreeData)
    >>> partitions = get_feature_partitions(data)
    >>> partition = next(partitions)
    >>> partition.decision_boundary == 0.7
    True
    >>> # Note that the first row comes after the second and third when sorted by feature value  # noqa
    >>> list(partition.partition)
    [False, True, True, False, False, False]
    >>> # |         | left | right |
    >>> # |---------+------+-------|
    >>> # | counts  |  2,0 | 2, 2  |
    >>> # | weight  |  2/6 | 4/6   |
    >>> # | entropy |    0 | 1     |
    >>> partition.score == 2/3
    True
    >>> partition = next(partitions)
    >>> partition.decision_boundary == 1.2
    True
    >>> list(partition.partition)
    [True, True, True, False, False, False]
    >>> # |         | left | right                              |
    >>> # |---------+------+------------------------------------|
    >>> # | counts  | 3,0  | 1, 2                               |
    >>> # | weight  | 3/6  | 3/6                                |
    >>> # | entropy | 0    | -(1/3)*log2(1/3) - (2/3)*log2(2/3) |
    >>> approximately_equal(partition.score, -(1/2)*( (1/3)*log2(1/3) + (2/3)*log2(2/3)))  # noqa
    True
    """
    n, d = data.shape
    assert n > 1 and d == 2
    counts = Counter(data.y)
    sort_permutation = data.X.ravel().argsort()
    data = data[sort_permutation, :]

    n, _ = data.shape

    left_counts = Counter()

    decision_boundary, _ = data[0]
    zero_feature_variance = True
    for i, (x, y) in enumerate(data):
        if x > decision_boundary:
            partition = np.ones(n, dtype=np.bool)
            partition[sort_permutation[i:]] = False
            score = weighted_average_entropy(left_counts, counts - left_counts)
            yield Partition(decision_boundary, partition, score)
            decision_boundary = x
            zero_feature_variance = False
        left_counts[y] += 1

    if zero_feature_variance:
        partition = np.ones(n, dtype=np.bool)
        score = weighted_average_entropy(left_counts, counts - left_counts)
        yield Partition(x, partition, score)


def weighted_average_entropy(*counters):
    size = lambda counter: sum(counter.values())
    n_total = sum(size(c) for c in counters)
    return sum(
        (size(c) * entropy(c.values()))
        for c in counters
    ) / n_total
