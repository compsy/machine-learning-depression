import numpy as np


class OutputDataCleaner:

    def clean(self, data, incorrect_indices):
        return np.delete(data, incorrect_indices, axis=0)

    def find_incomplete_rows(self, data):
        incorrect_indices = []
        index = 0  # TODO this can probably be done in one statement
        for row in data:
            # If a row contains NA's, add the pident to a list
            if np.any(np.isnan(row)):
                incorrect_indices.append(index)
            index += 1

        return incorrect_indices
