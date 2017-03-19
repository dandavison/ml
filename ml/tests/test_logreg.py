import numpy as np
from numpy import array

from ml.models.logistic_regression import LogisticRegressionWithL2Regularization  # noqa
from ml.tests import TestCase

class TestLogisticRegression(TestCase):

    def _get_data_1(self):
        """Observe 200 1D data points.

        100 have x=1. Of these 25 have Y=1 and 75 have Y=0.
        100 have x=0. Of these 50 have Y=1 and 50 have Y=0.

        logistic: R -> (0, 1)
        p = 1 / (1 + e^{-x.w})

        inverse logistic: (0, 1) -> R
        x.w = log p/(1-p)

        E.g. inverse_logistic(0.25) = log( (1/4)/(3/4) ) = -log 3

        The solution to this problem should predict:

        Probability at x=1 is 0.25, which is logistic(-log(3) * x + 0).
        Probability at x=0 is 0.50, which is logistic(-log(3) * x + 0).

        So the solution should be w=-log(3) ~= -1.1 with offset parameter = 0.
        """
        _Xy = np.vstack([
            [(0, 0)] * 50,
            [(0, 1)] * 50,
            [(1, 0)] * 75,
            [(1, 1)] * 25,
        ])
        X = _Xy[:,:-1]
        y = _Xy[:,[-1]]

        return X, y

    def _get_data_2(self):
        """
        Same as data set 1 but two dimensional. The second feature has no effect on
        the class probabilities.
        """
        _Xy = np.vstack([
            # Each row is (x1, x2, y)

            # Data points at x1 = 0
            #   x2 = 0
            [(0, 0, 0)] * 50,
            [(0, 0, 1)] * 50,
            #   x2 = 1
            [(0, 1, 0)] * 50,
            [(0, 1, 1)] * 50,

            # Data points at x1 = 1
            #   x2 = 0
            [(1, 0, 0)] * 75,
            [(1, 0, 1)] * 25,
            #   x2 = 1
            [(1, 1, 0)] * 75,
            [(1, 1, 1)] * 25,
        ])
        X = _Xy[:,:-1]
        y = _Xy[:,[-1]]

        return X, y

    def _get_data_3(self):
        """
        Like data set 2 but second feature does have an effect on
        the class probabilities.
        """
        _Xy = np.vstack([
            # Each row is (x1, x2, y)

            # Data points at x1 = 0
            #   x2 = 0
            [(0, 0, 0)] * 50,
            [(0, 0, 1)] * 50,
            #   x2 = 1
            [(0, 1, 0)] * 60,
            [(0, 1, 1)] * 40,

            # Data points at x1 = 1
            #   x2 = 0
            [(1, 0, 0)] * 75,
            [(1, 0, 1)] * 25,
            #   x2 = 1
            [(1, 1, 0)] * 90,
            [(1, 1, 1)] * 10,
        ])
        X = _Xy[:,:-1]
        y = _Xy[:,[-1]]

        return X, y

    def _run_test(self, X, y, add_offset=True):
        if add_offset:
            n, d = X.shape
            X = np.hstack([X, np.ones((n, 1))])

        epsilon = 0.01
        model = LogisticRegressionWithL2Regularization(
            _lambda=0,
            epsilon=epsilon,
        )

        w1 = model.fit(X, y, n_iter=75)
        w2 = model.fit(X, y, n_iter=200)

        self.assertLess(
            model.loglike(w1, X, y),
            model.loglike(w2, X, y),
        )

        return X, y, model, w1, w2

    def test_1_no_offset(self):
        X, y, model, w1, w2 = self._run_test(*self._get_data_1(),
                                             add_offset=False)

        self.assertTrue(
            w1.shape == w2.shape == (1, 1)
        )

        w_expected = -np.log(3)

        delta1 = np.abs(w1 - w_expected)
        delta2 = np.abs(w2 - w_expected)

        self.assertTrue((delta1 < 1e-6).all())
        self.assertTrue((delta2 < 1e-6).all())
        self.assertTrue((delta2 < delta1).all())

    def test_1(self):
        X, y, model, w1, w2 = self._run_test(*self._get_data_1())

        self.assertTrue(
            w1.shape == w2.shape == (2, 1)
        )

        w_expected = array([
            [-np.log(3)],
            [0],
        ])
        delta1 = np.abs(w1 - w_expected)
        delta2 = np.abs(w2 - w_expected)

        self.assertTrue((delta1 < 1e-3).all())
        self.assertTrue((delta2 < 1e-6).all())
        self.assertTrue((delta2 < delta1).all())

    def test_2(self):
        X, y, model, w1, w2 = self._run_test(*self._get_data_2())

        self.assertTrue(
            w1.shape == w2.shape == (3, 1)
        )

        w_expected = array([
            [-np.log(3)],
            [0],
            [0],
        ])
        delta1 = np.abs(w1 - w_expected)
        delta2 = np.abs(w2 - w_expected)

        self.assertTrue((delta1 < 1e-4).all())
        self.assertTrue((delta2 < 1e-6).all())
        self.assertTrue((delta2 < delta1).all())


    def test_3(self):
        X, y, model, w1, w2 = self._run_test(*self._get_data_3())

        n, d = X.shape
        X = np.hstack([X, np.ones((n, 1))])
        n, d = X.shape

        epsilon = 0.01
        model = LogisticRegressionWithL2Regularization(
            _lambda=0,
            epsilon=epsilon,
        )

        w1 = model.fit(X, y, n_iter=10)
        w2 = model.fit(X, y, n_iter=10)

        # TODO


if __name__ == '__main__':
    TestLogisticRegression().test_2()
    print('OK')
