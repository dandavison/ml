import json
import numpy as np

from ml.utils import mean
from ml.utils import random_partition


class Model:

    @property
    def name(self):
        return self.__class__.__name__

    def fit(self, X, y, **kwargs):
        """
        Fit model to inputs X and outcomes y.

        Return the model instance.
        """
        raise NotImplementedError

    def serialize(self):
        return json.dumps(self.model, sort_keys=True)


class Classifier(Model):

    def predict(self, X):
        raise NotImplementedError

    def get_error_rate(self,
                       validation_data,
                       validation_labels):

        predictions = self.predict(validation_data)
        return mean(r != y for r, y in zip(predictions.ravel(),
                                           validation_labels.ravel()))

    def get_error_rates(self,
                        training_data_subset_sizes,
                        training_data,
                        training_labels,
                        validation_data,
                        validation_labels):
        """
        Return estimated error rate for each training data subset size.

        Dimensions of array are (#{training_data_subsets}, 2)
        """
        error_rates = []
        for n_t in training_data_subset_sizes:
            training_data_subset, training_labels_subset, _, _ = (
                random_partition(training_data, training_labels, n_t))
            error_rates.append(
                self.get_error_rate(
                    training_data_subset,
                    training_labels_subset,
                    validation_data,
                    validation_labels,
                )
            )
        return np.array(list(zip(training_data_subset_sizes, error_rates)))
