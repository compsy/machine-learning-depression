import warnings

import numpy as np


class DataTransformer:

    def getVariableIndices(self, all_names, selected_names):
        variable_indices = []
        for name in selected_names:
            if (len(np.where(all_names == name)[0]) == 0):
                warnings.warn('Variable not found ' + name)
                continue
            variable_indices.append(np.where(all_names == name)[0][0])
        return variable_indices
