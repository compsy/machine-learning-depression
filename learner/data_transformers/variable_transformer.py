from _ctypes import ArgumentError
import warnings
from sklearn.preprocessing import PolynomialFeatures
from .data_transformer import DataTransformer
import numpy as np


class VariableTransformer(DataTransformer):
    def __init__(self, header):
        self.header = header

    def log_transform(self, x, variable_to_transform, additive=False):
        # Note that the X data might be used as 'passed by reference' and the values of the array may change

        var_loc = self.get_variable_index(self.header, variable_to_transform)
        values_to_add = x[:, var_loc].copy()
        if (var_loc is None):
            raise ValueError('Variable %s not found..' % variable_to_transform)

        # If the variable contains a 0 or lower, add the minimal value as a constant, so the log can succeed
        if (min(x[:, var_loc]) <= 0):
            adder = (min(x[:, var_loc]) * -1) + 1
            warnings.warn(
                '%s contains 0s, thats not allowed for log transforms. Adding a constant of %d.'
                % (variable_to_transform, adder))
            values_to_add += adder


        # if the result should be appended to the original dataframe
        if (additive):
            self.header = np.insert(self.header, var_loc + 1, 'log_' + variable_to_transform)
            x = np.insert(x, var_loc + 1, np.log(values_to_add), axis=1)
        else:
            x[:, var_loc] = np.log(values_to_add)

        return x

    def sqrt_transform(self, x, variable_to_transform, additive=False):
        # Note that the X data might be used as 'passed by reference' and the values of the array may change

        var_loc = self.get_variable_index(self.header, variable_to_transform)
        if (var_loc is None):
            raise ValueError('Variable %s not found..' % variable_to_transform)

        values_to_add = x[:, var_loc].copy()
        if (min(values_to_add) < 0):
            adder = (min(values_to_add) * -1) + 1
            warnings.warn(
                '%s contains 0s, thats not allowed for log transforms. Adding a constant of %d.'
                % (variable_to_transform, adder))
            values_to_add += adder

        if (additive):
            self.header = np.insert(self.header,var_loc+1, 'sqrt_' + variable_to_transform)
            x = np.insert(x, var_loc + 1, np.sqrt(values_to_add), axis=1)
        else:
            x[:, var_loc] = np.sqrt(values_to_add)

        return x
