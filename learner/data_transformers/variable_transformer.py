from _ctypes import ArgumentError
import warnings
from sklearn.preprocessing import PolynomialFeatures
from .data_transformer import DataTransformer
import numpy as np


class VariableTransformer(DataTransformer):

    def __init__(self, header):
        self.header = header

    def log_transform(self, x, variable_to_transform, additive=False):
        var_loc = self.get_variable_index(self.header, variable_to_transform)
        if (var_loc is None): raise ValueError('Variable %s not found..' % variable_to_transform)
        if (min(x[:, var_loc]) <= 0):
            adder = (min(x[:, var_loc]) * -1) + 1
            warnings.warn('%s contains 0s, thats not allowed for log transforms. Adding a constant of %d.' %
                          (variable_to_transform, adder))
            x[:, var_loc] += adder

        x[:, var_loc] = np.log(x[:, var_loc])

    def sqrt_transform(self, x, variable_to_transform, additive=False):
        var_loc = self.get_variable_index(self.header, variable_to_transform)
        if (var_loc is None): raise ValueError('Variable %s not found..' % variable_to_transform)

        values_to_add = x[:, var_loc].copy()
        if (min(values_to_add) < 0):
            adder = (min(values_to_add) * -1) + 1
            warnings.warn('%s contains 0s, thats not allowed for log transforms. Adding a constant of %d.' %
                          (variable_to_transform, adder))
            values_to_add += adder

        if (additive):
            self.header.append('sqrt_' + variable_to_transform)
            x[:, var_loc + 1] = np.sqrt(values_to_add)
        else:
            x[:, var_loc] = np.sqrt(values_to_add)
