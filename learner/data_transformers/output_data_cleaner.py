import numpy as np
from .data_transformer import DataTransformer
from data_output.std_logger import L


class OutputDataCleaner(DataTransformer):

    def clean(self, data, incorrect_indices):
        return np.delete(data, incorrect_indices, axis=0)

    def find_incomplete_rows(self, data, header, print_info=False):
        incorrect_indices = []
        missing_value_hash = {}
        missing_indices = {}
        index = 0  # TODO this can probably be done in one statement
        for row in data:
            # If a row contains NA's, add the pident to a list
            zippd = list(zip(header, np.isnan(row)))

            # Loop through the header
            for z in zippd:
                if z[1]:
                    key = z[0]
                    temp = missing_value_hash.get(key, 0)
                    missing_value_hash[key] = temp + 1

                    if not key in missing_indices:
                        missing_indices[key] = []

                    missing_indices[key].append(index)

            if np.any(np.isnan(row)):
                incorrect_indices.append(index)
            index += 1

        if print_info: L.info('The following keys have the most missings:')
        items = sorted(missing_value_hash.items(), key=lambda k_v: (k_v[1], k_v[0]), reverse=True)
        variables_that_are_gone = set()
        for key, value in items:
            current_indices = set(missing_indices[key])
            new_removed_vars = len(set(current_indices).difference(variables_that_are_gone))

            variables_that_are_gone = variables_that_are_gone | (
                set(current_indices).difference(variables_that_are_gone))

            if print_info: L.info("--> %s (of which %d are new) \t %s" % (value, new_removed_vars, key))
        return incorrect_indices
