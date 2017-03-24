from collections import Counter
from itertools import starmap

import numpy as np
import pygraphviz as pgv

from ml.models import Classifier
from ml.utils import entropy
from ml.utils import inverse_permutation
from ml.utils import mean
from ml.utils import starmin


VERBOSE = False


class DecisionTree(Classifier):

    def __init__(self,
                 feature_names=None,
                 label_names=None,
                 n_trees=1,
                 randomize_features=False,
                 randomize_observations=False,
                 max_depth=np.inf):

        assert (randomize_observations or randomize_features) != (n_trees == 1)

        self.trees = []
        self.feature_names = feature_names
        self.label_names = label_names
        self.model = {
            'n_trees': n_trees,
            'randomize_features': randomize_features,
            'randomize_observations': randomize_observations,
            'max_depth': max_depth,
        }

    def fit(self, X, y):
        n, d = X.shape
        row_indices = range(n)
        for i in range(self.model['n_trees']):
            if self.model['randomize_observations']:
                bootstrap_sample = np.random.choice(row_indices, n, replace=True)
                _X, _y = X[bootstrap_sample], y[bootstrap_sample]
            else:
                _X, _y = X, y
            data = np.hstack([X, y]).view(DecisionTreeData)
            tree = self._grow_tree(data, 0)
            self.trees.append(tree)
        return self

    def predict(self, X):
        n, d = X.shape
        y_pred = []
        for x in X:
            mode = (Counter(tree.predict(x) for tree in self.trees)
                    .most_common()[0][0])
            y_pred.append(mode)
        return np.array(y_pred).reshape((n, 1))

    def draw(self, filename):
        """
        Create an image of the tree using graphviz.
        """
        graph = pgv.AGraph(directed=True)
        for i, tree in enumerate(self.trees):
            tree.add_to_graph(graph)
            graph.layout('dot')
            graph.draw('%s%s' % (
                filename,
                ('-%d' % i) if len(self.trees) > 1 else ''))

    def describe(self):
        for tree in self.trees:
            tree.describe()

    def _grow_tree(self, data, depth):
        label_counts = Counter(data.y)
        labels = label_counts.keys()
        predict = lambda: label_counts.most_common()[0][0]
        if len(labels) == 1 or depth > self.model['max_depth']:
            node = self._node_factory(label=predict(), counts=label_counts)
        else:
            partition = choose_partition(data, self.model['randomize_features'])
            left, right = data[partition.partition, :], data[~partition.partition, :]
            n_right, d = right.shape
            if n_right == 0:
                # All features are constant on this subset of sample points
                node = self._node_factory(label=predict(), counts=label_counts)
            else:
                node = self._node_factory(self._grow_tree(left, depth + 1),
                                          self._grow_tree(right, depth + 1,),
                                          partition.feature,
                                          partition.decision_boundary,
                                          counts=label_counts)
        if VERBOSE:
            print(node, flush=True)

        return node

    def _node_factory(self, *args, **kwargs):
        return Node(*args, **kwargs, forest=self)


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
                 forest=None):

        self.left = left
        self.right = right
        self.feature = feature
        self.decision_boundary = decision_boundary
        self.label = label
        self.counts = counts
        self.forest = forest

    @property
    def is_leaf(self):
        return self.label is not None

    @property
    def feature_name(self):
        if self.feature is None:
            return ''
        elif self.forest.feature_names is not None:
            return self.forest.feature_names[self.feature]
        else:
            return 'feature %d' % self.feature

    @property
    def label_name(self):
        if self.label is None:
            return ''
        elif self.forest.label_names is None:
            return self.label
        else:
            return self.forest.label_names[int(self.label)]

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
                       label_name=self.label_name,
                       feature_name=self.feature_name)
        if self.is_leaf:
            return 'Node {self_id}: class {label_name}'.format(**context)
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
        def format_counts(counts):
            if self.forest.label_names:
                counts = {self.forest.label_names[int(k)]: n for k, n in counts.items()}
            return (
                '{%s}' % ', '.join(starmap('{}:{}'.format, sorted(counts.items()))))
        if self.is_leaf:
            label = self.label_name
        else:
            label = '{feature_name} > {decision_boundary}?'.format(
                feature_name=self.feature_name,
                **vars(self))
        graph.add_node(self, label=label)
        if self.left:
            self.left.add_to_graph(graph)
            graph.add_edge(self, self.left, label='No\n%s' % format_counts(self.left.counts))
        if self.right:
            self.right.add_to_graph(graph)
            graph.add_edge(self, self.right, label='Yes\n%s' % format_counts(self.right.counts))


class Partition:
    def __init__(self, decision_boundary, partition, score, feature=None):
        self.decision_boundary = decision_boundary
        self.partition = partition
        self.score = score
        self.feature = feature


def choose_partition(data, randomize_features):
    if not randomize_features:
        return _choose_partition(data)
    else:
        # Partition the features into non-overlapping subsets each of size
        # n_features, with random uniform allocation. Start with the first such
        # subset. Find the best partition of the observations when restricted
        # to this subset of features. If this is not a trvial partition
        # (i.e. trivial means all features in the subset were constant on this
        # subset of the observations) then short circuit and return this
        # partition. Otherwise, continue to the next subset of features.
        n, d = data.X.shape
        features = np.arange(d)
        feature_permutation = np.random.choice(features, replace=False, size=d)
        n_features = int(np.sqrt(d))
        offset = 0
        while True:
            feature_subset = features[feature_permutation[offset:(offset + n_features)]]
            if not len(feature_subset):
                return partition
            else:
                partition = _choose_partition(data[:, feature_subset])
                if not partition.partition.all():
                    return partition
                else:
                    offset += n_features

def _choose_partition(data):
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
