import numpy as np


class OutputDataFrameCleaner:

    def clean(self, data, selected_variables, all_names):
        variable_indices = self.getVariableIndices(all_names, selected_variables)
        used_data = data[:, variable_indices]
        incorrect_indices = []
        index = 0  # TODO this can probably be done in one statement
        for row in used_data:
            # If a row contains NA's, add the pident to a list
            if np.any(np.isnan(row)):
                incorrect_indices.append(index)
            index += 1

        return np.delete(used_data, incorrect_indices, axis=0)


    def getVariableIndices(self, all_names, selected_names):
        variable_indices = []
        for name in selected_names:
            variable_indices.append(np.where(all_names == name)[0][0])
        return variable_indices