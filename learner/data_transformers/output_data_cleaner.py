import numpy as np
from .data_transformer import DataTransformer


class OutputDataCleaner(DataTransformer):

    def clean(self, data, incorrect_indices):
        return np.delete(data, incorrect_indices, axis=0)

    def find_incomplete_rows(self, data, header):
        incorrect_indices = []
        missing_value_hash = {}
        index = 0  # TODO this can probably be done in one statement
        for row in data:
            # If a row contains NA's, add the pident to a list
            zippd = list(zip(header, np.isnan(row)))
            for z in zippd:
                if z[1]:
                    key = z[0]
                    temp = missing_value_hash.get(key, 0)
                    missing_value_hash[key] = temp + 1

            if np.any(np.isnan(row)):
                incorrect_indices.append(index)
            index += 1

        print('\t -> The following keys have the most missings:')
        for key, value in sorted(missing_value_hash.items(), key=lambda k_v: (k_v[1], k_v[0]), reverse=True):
            print("\t --> %s: \t\t %s" % (key, value))
        return incorrect_indices
