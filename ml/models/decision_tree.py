from collections import Counter


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

    def __init__(self, left=None, right=None, feature=None, _class=None):
        self.left = None
        self.right = None
        self.feature = feature
        self._class = _class


def grow_tree(data):
    classes = set(y)
    if len(classes) == 1:
        [c] = classes
        return Node(_class=c)
    else:
        left, right, feature = choose_split(data)
        return Node(grow_tree(left),
                    grow_tree(right),
                    feature)


def choose_split(data):
    n, d = data.X.shape
    class_counts = Counter(data.y)
    split, j, score = max(
        (split, j, score)
        for j in range(d)
        for (split, score) in get_splits_on_feature(data[:, [j, -1]], class_counts),
        key=lambda (split, j, score): score)
    left, right = data[split, :], data[~split, :]
    return left, right, j


def get_splits_on_feature(data, class_counts):
    permutation = data[:, 0].argsort()
    data = data[permutation, :]

    left_counts = Counter()

    curr, _ = data[0]
    for i, (x, y) in enumerate(data):
        if x > curr:
            yield permutation[]
        else:
            left_counts[y] += 1




def get_feature_split(data):



def weighted_average_entropy(datasets):
    n = [d.shape[0] for d in datasets]
