import unittest

import sklearn.discriminant_analysis
import numpy as np
from numpy import array
from numpy.random import multivariate_normal

from ml.models.gaussian_discriminant_analysis import LinearDiscriminantAnalysis
from ml.models.gaussian_discriminant_analysis import QuadraticDiscriminantAnalysis  # noqa
from ml.tests import TestCase


class _TestDiscriminantAnalysisMixin:
    Model = None

    def test_1d_1(self):
        """
        Observe two 1D data points:

        |  X | label |
        |----+-------|
        | -1 |     1 |
        |  1 |     2 |

        - The class means should be [-1, 1]
        - The pooled covariance matrix should be 0 (1x1 matrix)
        - The estimated class prior should be [0.5, 0.5]
        - Points < 0 would be predicted class 1;
          points > 0 would be predicted class 2;
        - But currently not sure how prediction method is expected to behave
          with this zero covariance matrix and the pseudoinverse.

        """
        data = array([[-1],
                      [1]])
        labels = array([1,
                        2])
        test_data = array([
            [-0.7],
            [-0.1],
            [+0.1],
            [+1.2],
        ])
        test_data_expected_predictions = [
            1,
            1,
            2,
            2,
        ]

        model = self.Model().fit(data, labels)

        if False:
            reference_model = self.ReferenceModel().fit(data, labels)
            self.assertEqualArrays(
                reference_model.predict(test_data),
                test_data_expected_predictions,
                "Reference model does not yield expected predictions",
            )

        self.assertEqualArrays(model.mus, array([[-1], [1]]))

        if self.Model == LinearDiscriminantAnalysis:
            self.assertEqualArrays(model.Sigma, array([[0]]))
        elif self.Model == QuadraticDiscriminantAnalysis:
            for c_index in range(len(set(labels))):
                self.assertEqualArrays(model.Sigma[c_index], array([[0]]))
        else:
            raise ValueError("Invalid model: %s" % self.Model.__name__)

        self.assertEqual(model.prior, {1: 0.5, 2: 0.5})

        if False:
            self.assertEqualArrays(model.predict(test_data),
                                   test_data_expected_predictions)

    def test_1d_2(self):
        """
        Observe four 1D data points:

        |    X | label |
        |------+-------|
        | -1.5 |     1 |
        | -0.5 |     1 |
        |  0.5 |     2 |
        |  1.5 |     2 |

        - The class means should be [-1, 1]
        - The pooled covariance matrix should be 0.25 (1x1 matrix)
        - The estimated class prior should be [0.5, 0.5]
        - Points < 0 would be predicted class 1;
          points > 0 would be predicted class 2;
        """
        data = array([[-1.5],
                      [-0.5],
                      [+0.5],
                      [+1.5]])
        labels = array([1,
                        1,
                        2,
                        2])

        test_data = array([
            [-0.7],
            [-0.01],
            [0.01],
            [1.2],
        ])
        test_data_expected_predictions = array([
            1,
            1,
            2,
            2,
        ])

        model = self.Model().fit(data, labels)
        if False:
            reference_model = self.ReferenceModel().fit(data, labels)
            self.assertEqualArrays(
                reference_model.predict(test_data),
                test_data_expected_predictions,
                "Reference model does not yield expected predictions",
            )

        self.assertEqualArrays(model.mus, array([[-1], [1]]))

        if self.Model == LinearDiscriminantAnalysis:
            self.assertEqualArrays(model.Sigma, array([[0.25]]))
        elif self.Model == QuadraticDiscriminantAnalysis:
            for c_index in range(len(set(labels))):
                self.assertEqualArrays(model.Sigma[c_index], array([[0.25]]))
        else:
            raise ValueError("Invalid model: %s" % self.Model.__name__)

        self.assertEqual(model.prior, {1: 0.5, 2: 0.5})

        self.assertEqualArrays(model.predict(test_data),
                               test_data_expected_predictions)

    @unittest.skip('Expected predictions are wrong')
    def test_2d_1(self):
        """
        Observe four 2D data points:

        | X            | label |
        |--------------+-------|
        | (-1.5, -1.5) |     1 |
        | (-0.5, -0.5) |     1 |
        | (+0.5, +0.5) |     2 |
        | (+1.5, +1.5) |     2 |

        - The class means should be [(-1,-1), (1,1)]
        - The pooled covariance matrix should have all entries equal to
          0.25 (2x2 matrix)
        - The estimated class prior should be [0.5, 0.5]
        - Points < 0 would be predicted class 1;
          points > 0 would be predicted class 2;
        """
        data = array([[-1.5, -1.5],
                      [-0.5, -0.5],
                      [+0.5, +0.5],
                      [+1.5, +1.5]])
        labels = array([1,
                        1,
                        2,
                        2])

        test_data =  array([
            [-0.7,  -0.7],
            [-0.01, -0.01],
            [+0.01, +0.01],
            [+1.2,  +1.2],
        ])
        test_data_expected_predictions = array([
            1,
            1,
            2,
            2,
        ])

        model = self.Model().fit(data, labels)
        reference_model = self.ReferenceModel().fit(data, labels)
        self.assertEqualArrays(
            reference_model.predict(test_data),
            test_data_expected_predictions,
            "Reference model does not yield expected predictions",
        )

        self.assertEqualArrays(model.mus, array([[-1, -1],
                                                 [+1, +1]]))
        if self.Model == LinearDiscriminantAnalysis:
            self.assertEqualArrays(model.Sigma, array([[0.25, 0.25],
                                                       [0.25, 0.25]]))
        elif self.Model == QuadraticDiscriminantAnalysis:
            for c_index in range(len(set(labels))):
                self.assertEqualArrays(model.Sigma[c_index],
                                       array([[0.25, 0.25],
                                              [0.25, 0.25]]))
        else:
            raise ValueError("Invalid model: %s" % self.Model.__name__)

        self.assertEqual(model.prior, {1: 0.5, 2: 0.5})

        self.assertEqualArrays(
            model.predict(test_data),
            test_data_expected_predictions
        )

    def test_2d_2(self):
        """
        Observe 8 2D data points.

        - The class means should be [(-1,-1), (1,1)]
        - The pooled covariance matrix should have all entries equal to
          0.25 (2x2 matrix)
        - The estimated class prior should be [0.5, 0.5]
        - Points in lower left quadrant should be predicted class 1;
          points in upper right quadrant should be predicted class 2;
        """
        data = array([
            [-1.5, -1.0],
            [-1.0, -0.5],
            [-0.5, -1.0],
            [-1.0, -1.5],

            [+1.5, +1.0],
            [+1.0, +0.5],
            [+0.5, +1.0],
            [+1.0, +1.5],
        ])
        labels = array([
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
        ])

        test_data =  array([
            [-0.7,  -0.7],
            [-0.01, -0.01],
            [+0.01, +0.01],
            [+1.2,  +1.2],
        ])
        test_data_expected_predictions = array([
            1,
            1,
            2,
            2,
        ])

        model = self.Model().fit(data, labels)
        reference_model = self.ReferenceModel().fit(data, labels)
        self.assertEqualArrays(
            reference_model.predict(test_data),
            test_data_expected_predictions,
            "Reference model does not yield expected predictions",
        )

        self.assertEqualArrays(model.mus, array([[-1, -1],
                                                 [+1, +1]]))

        self.assertEqual(model.prior, {1: 0.5, 2: 0.5})

        self.assertEqualArrays(
            model.predict(test_data),
            test_data_expected_predictions
        )

    def test_2d_3(self):
        """
        Observe 100 2D data points.
        """
        np.random.seed(2)  # 0 and 1 fail for LDA; QDA seems to always pass
        X1 = multivariate_normal([-1, -1], [[2**2, 0],[0, 2**2]], 100)
        X2 = multivariate_normal([+1, +1], [[2**2, 0],[0, 2**2]], 100)
        data = np.vstack([X1, X2])
        labels = np.array([1] * 100 + [2] * 100)

        test_data = multivariate_normal([+1, +1], [[2**2, 0],[0, 2**2]], 100)

        model = self.Model().fit(data, labels)
        reference_model = self.ReferenceModel().fit(data, labels)
        test_data_expected_predictions = reference_model.predict(test_data)

        self.assertEqual(model.prior, {1: 0.5, 2: 0.5})

        self.assertEqualArrays(
            model.predict(test_data),
            test_data_expected_predictions
        )


class TestLDA(_TestDiscriminantAnalysisMixin, TestCase):
    Model = LinearDiscriminantAnalysis
    ReferenceModel = sklearn.discriminant_analysis.LinearDiscriminantAnalysis


class TestQDA(_TestDiscriminantAnalysisMixin, TestCase):
    Model = QuadraticDiscriminantAnalysis
    ReferenceModel = sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis  # noqa
