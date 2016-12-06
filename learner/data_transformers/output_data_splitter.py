from .data_transformer import DataTransformer


class OutputDataSplitter(DataTransformer):
    def split(self, data, variable_set, selected_variables):
        """ Splits the selected variables from the data
        :params:
        - data: the original dataframe
        - variable_set: the variables in the original dataframe
        - selected_variables: the variables we'd like to have

        :returns: the dataframe with only the selected variables

        """
        variable_indices = self.get_variable_indices(variable_set, selected_variables)
        return data[:, variable_indices]
