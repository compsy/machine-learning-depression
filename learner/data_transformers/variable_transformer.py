from _ctypes import ArgumentError
import warnings
from sklearn.preprocessing import PolynomialFeatures
from .data_transformer import DataTransformer
import numpy as np


class VariableTransformer(DataTransformer):

    def __init__(self, header):
        self.header = header

    def log_transform(self, x, variable_to_transform):
        var_loc = self.get_variable_index(self.header, variable_to_transform)
        if (var_loc is None): raise ValueError('Variable %s not found..' % variable_to_transform)
        if (min(x[:, var_loc]) <= 0):
            adder = (min(x[:, var_loc]) * -1) + 1
            warnings.warn('%s contains 0s, thats not allowed for log transforms. Adding a constant of %d.' %
                          (variable_to_transform, adder))
            x[:, var_loc] += adder

        x[:, var_loc] = np.log(x[:, var_loc])

    def sqrt_transform(self, x, variable_to_transform):
        var_loc = self.get_variable_index(self.header, variable_to_transform)
        if (var_loc is None): raise ValueError('Variable %s not found..' % variable_to_transform)

        if (min(x[:, var_loc]) < 0):
            adder = (min(x[:, var_loc]) * -1) + 1
            warnings.warn('%s contains 0s, thats not allowed for log transforms. Adding a constant of %d.' %
                          (variable_to_transform, adder))
            x[:, var_loc] += adder
        x[:, var_loc] = np.sqrt(x[:, var_loc])
