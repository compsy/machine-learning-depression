from .data_transformer import DataTransformer


class OutputDataSplitter(DataTransformer):

    def split(self, data, variable_set, selected_variables):
        variable_indices = self.getVariableIndices(variable_set,
                                                   selected_variables)
        return data[:, variable_indices]
