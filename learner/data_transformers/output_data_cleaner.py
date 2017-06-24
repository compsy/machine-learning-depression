import numpy as np
from learner.data_transformers.data_transformer import DataTransformer
from learner.data_output.std_logger import L


class OutputDataCleaner(DataTransformer):

    @staticmethod
    def clean(data, incorrect_indices):
        """
        Clean the data based on the incorrect indices
        """
        return data.drop(incorrect_indices)

    @staticmethod
    def find_incomplete_rows(data, header, print_info=True):
        """
        Finds the rows that are incomplete and counts why they are incomplete.

        Parameters
        ----------
        data: the full dataset (pandas DF)
        header: list, the header of the elements to use
        print_info: boolean, whether to print the info or not
        """
        incorrect_indices = []
        missing_value_hash = {}
        missing_indices = {}
        for index, row in data.iterrows():

            # If a row contains NA's, add the pident to a list
            zippd = zip(header, np.isnan(row))

            # zippd is the header name + true or false for the value in this row whether it is nan.
            # Loop through the header
            for key_and_nan in zippd:
                # If z[1] is NaN
                if key_and_nan[1]:
                    key = key_and_nan[0]

                    # Store the stats, for this key there is a missing.
                    temp = missing_value_hash.get(key, 0)
                    missing_value_hash[key] = temp + 1

                    # Also store the index which is missing
                    missing_indices[key] = [] if not key in missing_indices else missing_indices[key]
                    missing_indices[key].append(index)

            # Add the index of the row to remove
            if np.any(np.isnan(row)):
                incorrect_indices.append(index)

        # Print statistics
        if print_info:
            L.info('The following keys have the most missings:')

        items = sorted(missing_value_hash.items(), key=lambda k_v: (k_v[1], k_v[0]), reverse=True)
        variables_that_are_gone = set()
        for key, value in items:
            current_indices = set(missing_indices[key])
            new_removed_vars = len(set(current_indices).difference(variables_that_are_gone))

            variables_that_are_gone = variables_that_are_gone | (
                set(current_indices).difference(variables_that_are_gone))

            if print_info:
                L.info("--> %s (of which %d are new) \t %s" % (value, new_removed_vars, key))

        return incorrect_indices
