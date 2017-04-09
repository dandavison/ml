from random import shuffle

import numpy as np


class LabeledData(np.ndarray):
    """
    A class representing training data with output values (labels).

    Must be instantiated with an array of dimension (n, d+1),
    the last column containing the output value y for the row.
    """
    @property
    def X(self):
        return self[:, :-1]

    @property
    def y(self):
        return self[:, -1]

    def random_partition(self, n_partition):
        """
        Split into n rows and remaining rows, after shuffling.
        """
        n, d = self.X.shape
        row_indices = list(range(n))
        shuffle(row_indices)
        return (
            LabeledData(
                self[row_indices[:n_partition]]
            ),
            LabeledData(
                self[row_indices[n_partition:]]
            ))
