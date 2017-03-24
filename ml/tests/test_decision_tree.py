from collections import Counter

import numpy as np
from numpy import array
from numpy.random import multivariate_normal

from ml.models.decision_tree import DecisionTree
from ml.tests import TestCase


class TestDecisionTree(TestCase):

    def test_1(self):
        for d in [1, 2]:
            X_train = array([
                [0] * d,
                [1] * d,
            ])
            y_train = array([
                [0],
                [1],
            ])
            X_test = array([
                [0] * d,
                [1] * d,
            ])
            y_test_expected = array([
                [0],
                [1],
            ])
        decision_tree = DecisionTree(label_names=[0, 1])
        decision_tree.fit(X_train, y_train)
        tree, = decision_tree.trees
        left, right = tree.left, tree.right
        self.assertTrue(tree.feature == 0)
        self.assertTrue(tree.decision_boundary == 0)
        self.assertTrue(left.is_leaf and right.is_leaf)
        self.assertEqual(
            Counter([left.label, right.label]),
            Counter([0, 1]),
        )
        self.assertEqualArrays(decision_tree.predict(X_test), y_test_expected)
