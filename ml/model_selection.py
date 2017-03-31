import random
from itertools import product

import pandas as pd

from ml.utils import mean
from ml.utils import random_partition


def get_accuracies(X, y, models, n_training, n_validation, n_replicates=1,
                    fixed_partition=False):
    """
    Return error rates computed on Cartesian product of models and training sizes.
    """
    assert not isinstance(n_training, int), \
        "n_training should be an iterable of training sizes"

    n, d = X.shape
    assert y.shape == (n, 1)
    assert max(n_training) + n_validation <= n

    X_validation, y_validation, X_train, y_train = random_partition(X, y, n_validation)

    if fixed_partition:
        rng_state = random.getstate()

    rows = []
    for model, n_train in product(models, n_training):
        print(model.serialize(), n_train, n_replicates)
        training_accuracies = []
        validation_accuracies = []
        for rep in range(n_replicates):

            if fixed_partition:
                random.setstate(rng_state)

            _X_train, _y_train, _, _ = random_partition(X_train, y_train, n_train)
            model.fit(_X_train, _y_train)
            training_accuracies.append(
                model.get_accuracy(X_train, y_train)
            )
            validation_accuracies.append(
                model.get_accuracy(X_validation, y_validation)
            )
        training_accuracy = mean(training_accuracies)
        validation_accuracy = mean(validation_accuracies)
        rows.append({
            'model': model.serialize(),
            'n_train': n_train,
            'training_accuracy': training_accuracy,
            'validation_accuracy': validation_accuracy,
        })
        print(training_accuracy, validation_accuracy, "\n")

    return pd.DataFrame(rows, columns=['model', 'n_train', 'training_accuracy', 'validation_accuracy'])
