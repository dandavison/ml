from itertools import product
from multiprocessing import Pool

import pandas as pd

from ml.utils import mean
from ml.utils import random_partition


def _get_error_rate(args):
    model, X_train, y_train, n_train, X_test, y_test = args
    _X_train, _y_train, _, _ = random_partition(X_train, y_train, n_train)
    model.fit(_X_train, _y_train)
    return model.get_error_rate(X_test, y_test)


def get_error_rates(X, y, models, n_training, n_validation, n_replicates=1):
    """
    Return error rates computed on Cartesian product of models and training sizes.
    """
    assert not isinstance(n_training, int), \
        "n_training should be an iterable of training sizes"

    n, d = X.shape
    assert y.shape == (n, 1)
    assert max(n_training) + n_validation <= n

    X_validation, y_validation, X_train, y_train = random_partition(X, y, n_validation)

    rows = []
    for model, n_train in product(models, n_training):
        args = model, X_train, y_train, n_train, X_validation, y_validation
        error_rate = mean(Pool().map(_get_error_rate, [args] * n_replicates))
        rows.append({
            'model': model.serialize(),
            'n_train': n_train,
            'error_rate': error_rate,
        })
        print(model.serialize(), n_train, error_rate)

    return pd.DataFrame(rows, columns=['model', 'n_train', 'error_rate'])
