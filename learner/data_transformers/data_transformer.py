import warnings

import numpy as np


class DataTransformer:

    def getVariableIndices(self, all_names, selected_names):
        variable_indices = []
        for name in selected_names:
            var_name = self.get_variable_index(all_names, name)
            if (var_name is not None): variable_indices.append(var_name)
        return variable_indices

    def get_variable_index(self, all_names, name):
        if (not isinstance(all_names, np.ndarray)):
            warnings.warn('Note that this function only works on NP arrays!')

        if (len(np.where(all_names == name)[0]) == 0):
            warnings.warn('Variable not found ' + name)
            return None

        return np.where(all_names == name)[0][0]
